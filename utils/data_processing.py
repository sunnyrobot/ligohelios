import numpy as np
import streamlit as st
from scipy import signal
import os
try:
    import healpy as hp
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from ligo.skymap import postprocess
    SKYMAP_AVAILABLE = True
except ImportError:
    SKYMAP_AVAILABLE = False

class DataProcessor:
    def __init__(self):
        self.data = {}
        self.processing_info = {}
        self.has_loaded_data = False
    
    def load_data(self, gps_time, duration, detectors, apply_bandpass=True, 
                  low_freq=35, high_freq=350, apply_whitening=False):
        """Load gravitational wave data from GWOSC"""
        try:
            # Try to import gwpy
            try:
                from gwpy.timeseries import TimeSeries
            except ImportError:
                st.error("gwpy library not found. Please install it to access LIGO data.")
                return False
            
            self.data = {}
            self.processing_info = {
                "GPS Time": gps_time,
                "Duration": f"{duration} seconds",
                "Detectors": ", ".join(detectors)
            }
            
            # Calculate time window
            start_time = gps_time - duration/2
            end_time = gps_time + duration/2
            
            for detector in detectors:
                try:
                    # Download strain data from GWOSC
                    strain = TimeSeries.fetch_open_data(
                        detector, start_time, end_time,
                        sample_rate=4096, cache=True
                    )
                    
                    # Convert to numpy arrays
                    time_array = strain.times.value
                    strain_array = strain.value
                    
                    # Apply signal processing
                    if apply_bandpass and low_freq and high_freq:
                        strain_array = self._apply_bandpass_filter(
                            strain_array, strain.sample_rate.value, low_freq, high_freq
                        )
                        self.processing_info["Bandpass Filter"] = f"{low_freq}-{high_freq} Hz"
                    
                    if apply_whitening:
                        strain_array = self._apply_whitening(strain_array, strain.sample_rate.value)
                        self.processing_info["Whitening"] = "Applied"
                    
                    # Store data
                    self.data[detector] = {
                        'time': time_array,
                        'strain': strain_array,
                        'sample_rate': strain.sample_rate.value,
                        'gps_start': start_time,
                        'gps_end': end_time
                    }
                    
                except Exception as e:
                    st.error(f"Failed to load data for {detector}: {str(e)}")
                    continue
            
            if self.data:
                self.has_loaded_data = True
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _apply_bandpass_filter(self, strain, sample_rate, low_freq, high_freq):
        """Apply bandpass filter to strain data"""
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_strain = signal.filtfilt(b, a, strain)
        return filtered_strain
    
    def _apply_whitening(self, strain, sample_rate, fft_length=4):
        """Apply whitening to reduce noise"""
        # Calculate ASD (Amplitude Spectral Density)
        freqs, psd = signal.welch(
            strain, fs=sample_rate, 
            nperseg=int(fft_length * sample_rate)
        )
        
        # Avoid division by zero
        psd[psd == 0] = np.inf
        
        # Whiten in frequency domain
        strain_fft = np.fft.fft(strain)
        freqs_fft = np.fft.fftfreq(len(strain), 1/sample_rate)
        
        # Interpolate PSD to match FFT frequencies
        psd_interp = np.interp(np.abs(freqs_fft), freqs, psd)
        
        # Whiten
        strain_fft_white = strain_fft / np.sqrt(psd_interp)
        strain_white = np.real(np.fft.ifft(strain_fft_white))
        
        return strain_white
    
    def calculate_cross_correlation(self, strain1, strain2):
        """Calculate cross-correlation between two strain signals"""
        # Ensure signals have the same length
        min_len = min(len(strain1), len(strain2))
        strain1 = strain1[:min_len]
        strain2 = strain2[:min_len]
        
        # Calculate cross-correlation
        correlation = signal.correlate(strain1, strain2, mode='full')
        correlation = correlation / np.max(np.abs(correlation))
        
        return correlation
    
    def has_data(self):
        """Check if data has been loaded"""
        return self.has_loaded_data and bool(self.data)
    
    def get_data(self):
        """Get loaded data"""
        return self.data
    
    def get_processing_info(self):
        """Get processing information"""
        return self.processing_info
    
    def calculate_snr(self, strain, template=None):
        """Calculate Signal-to-Noise Ratio"""
        if template is None:
            # Simple SNR estimate using peak vs RMS
            peak = np.max(np.abs(strain))
            rms = np.sqrt(np.mean(strain**2))
            return peak / rms if rms > 0 else 0
        else:
            # Matched filter SNR (simplified)
            correlation = signal.correlate(strain, template, mode='valid')
            return np.max(np.abs(correlation))
    
    def generate_gwb_skymap(self, map_type="snr", gwb_model="isotropic", nside=64):
        """Generate GWB (Gravitational Wave Background) sky map"""
        if not SKYMAP_AVAILABLE:
            st.error("Sky mapping libraries not available. Please install healpy and ligo.skymap.")
            return None
        
        try:
            npix = hp.nside2npix(nside)
            
            if map_type == "snr":
                # Generate SNR map based on GWB model
                if gwb_model == "isotropic":
                    # Isotropic GWB - uniform across sky
                    skymap = np.random.normal(0, 1, npix) + 5.0
                elif gwb_model == "dipole":
                    # Dipole anisotropy
                    theta, phi = hp.pix2ang(nside, range(npix))
                    skymap = 5.0 + 2.0 * np.cos(theta) + np.random.normal(0, 0.5, npix)
                elif gwb_model == "quadrupole":
                    # Quadrupole anisotropy
                    theta, phi = hp.pix2ang(nside, range(npix))
                    skymap = 5.0 + 1.5 * (3 * np.cos(theta)**2 - 1) + np.random.normal(0, 0.5, npix)
                elif gwb_model == "galactic":
                    # Galactic plane enhancement
                    theta, phi = hp.pix2ang(nside, range(npix))
                    b = np.pi/2 - theta  # Galactic latitude
                    skymap = 5.0 + 3.0 * np.exp(-np.abs(b)/(10*np.pi/180)) + np.random.normal(0, 0.5, npix)
                
            elif map_type == "upper_limit":
                # Generate upper limit map
                theta, phi = hp.pix2ang(nside, range(npix))
                # Simulate varying sensitivity across sky
                skymap = np.abs(np.random.normal(1e-15, 2e-16, npix))
                
            elif map_type == "detection_prob":
                # Detection probability map
                skymap = np.random.beta(2, 8, npix)  # Beta distribution for probabilities
            
            # Apply smoothing
            skymap = hp.smoothing(skymap, fwhm=np.pi/32)
            
            return {
                'map_data': skymap,
                'nside': nside,
                'map_type': map_type,
                'gwb_model': gwb_model,
                'npix': npix
            }
            
        except Exception as e:
            st.error(f"Error generating sky map: {str(e)}")
            return None
    
    def get_map_coordinates(self, skymap_data, coord_system="equatorial"):
        """Get coordinates for sky map visualization"""
        if not SKYMAP_AVAILABLE or skymap_data is None:
            return None
        
        nside = skymap_data['nside']
        npix = skymap_data['npix']
        
        # Get pixel coordinates
        theta, phi = hp.pix2ang(nside, range(npix))
        
        if coord_system == "equatorial":
            # Convert to RA/Dec
            ra = phi * 180 / np.pi
            dec = 90 - theta * 180 / np.pi
            return ra, dec
        elif coord_system == "galactic":
            # Convert to galactic coordinates
            c = SkyCoord(ra=phi*u.radian, dec=(np.pi/2-theta)*u.radian, frame='icrs')
            gal = c.galactic
            return gal.l.degree, gal.b.degree
        else:
            return phi * 180 / np.pi, 90 - theta * 180 / np.pi
    
    def manipulate_map_data(self, skymap_data, operation, **kwargs):
        """Perform on-the-fly data manipulations"""
        if skymap_data is None:
            return None
        
        map_data = skymap_data['map_data'].copy()
        
        if operation == "smooth":
            fwhm = kwargs.get('fwhm', np.pi/32)
            map_data = hp.smoothing(map_data, fwhm=fwhm)
        elif operation == "threshold":
            threshold = kwargs.get('threshold', 0)
            map_data = np.where(map_data > threshold, map_data, 0)
        elif operation == "log_scale":
            map_data = np.log10(np.abs(map_data) + 1e-20)
        elif operation == "normalize":
            map_data = (map_data - np.min(map_data)) / (np.max(map_data) - np.min(map_data))
        elif operation == "mask_poles":
            nside = skymap_data['nside']
            theta, phi = hp.pix2ang(nside, range(len(map_data)))
            mask = (theta > np.pi/6) & (theta < 5*np.pi/6)  # Exclude poles
            map_data = np.where(mask, map_data, hp.UNSEEN)
        
        result = skymap_data.copy()
        result['map_data'] = map_data
        return result
