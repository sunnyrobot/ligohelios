import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy import signal
import streamlit as st
try:
    import healpy as hp
    import matplotlib.pyplot as plt
    from matplotlib.projections import get_projection_class
    SKYMAP_AVAILABLE = True
except ImportError:
    SKYMAP_AVAILABLE = False

class GWVisualizer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_strain_timeseries(self, time, strain, detector_name):
        """Create interactive time series plot of strain data"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time,
            y=strain,
            mode='lines',
            name=f'{detector_name} Strain',
            line=dict(width=1, color=self.colors[0]),
            hovertemplate='<b>Time</b>: %{x:.3f} s<br>' +
                         '<b>Strain</b>: %{y:.2e}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{detector_name} Gravitational Wave Strain Data',
            xaxis_title='Time (s)',
            yaxis_title='Strain',
            hovermode='x unified',
            showlegend=True,
            height=400,
            template='plotly_white'
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def plot_spectrogram(self, time, strain, detector_name):
        """Create spectrogram plot"""
        # Calculate spectrogram
        sample_rate = len(strain) / (time[-1] - time[0])
        frequencies, times, Sxx = signal.spectrogram(
            strain, fs=sample_rate, nperseg=512, noverlap=256
        )
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        
        # Create plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=Sxx_db,
            x=times + time[0],  # Adjust time offset
            y=frequencies,
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)"),
            hovertemplate='<b>Time</b>: %{x:.3f} s<br>' +
                         '<b>Frequency</b>: %{y:.1f} Hz<br>' +
                         '<b>Power</b>: %{z:.1f} dB<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{detector_name} Spectrogram',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            height=400,
            template='plotly_white'
        )
        
        # Limit frequency range for better visualization
        fig.update_yaxis(range=[0, 500])
        
        return fig
    
    def plot_multi_detector_comparison(self, data):
        """Create comparison plot for multiple detectors"""
        fig = make_subplots(
            rows=len(data), cols=1,
            subplot_titles=[f'{detector} Detector' for detector in data.keys()],
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        for i, (detector, detector_data) in enumerate(data.items()):
            fig.add_trace(
                go.Scatter(
                    x=detector_data['time'],
                    y=detector_data['strain'],
                    mode='lines',
                    name=detector,
                    line=dict(width=1, color=self.colors[i % len(self.colors)]),
                    hovertemplate='<b>Time</b>: %{x:.3f} s<br>' +
                                 '<b>Strain</b>: %{y:.2e}<br>' +
                                 '<extra></extra>'
                ),
                row=i+1, col=1
            )
            
            # Add zero line for each subplot
            fig.add_hline(
                y=0, line_dash="dash", line_color="gray", 
                opacity=0.5, row=i+1, col=1
            )
        
        fig.update_layout(
            title='Multi-Detector Gravitational Wave Comparison',
            height=200 * len(data) + 100,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Time (s)", row=len(data), col=1)
        
        for i in range(len(data)):
            fig.update_yaxes(title_text="Strain", row=i+1, col=1)
        
        return fig
    
    def plot_frequency_domain(self, time, strain, detector_name):
        """Create frequency domain plot (ASD)"""
        sample_rate = len(strain) / (time[-1] - time[0])
        
        # Calculate ASD (Amplitude Spectral Density)
        frequencies, psd = signal.welch(
            strain, fs=sample_rate, nperseg=4096
        )
        asd = np.sqrt(psd)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=asd,
            mode='lines',
            name=f'{detector_name} ASD',
            line=dict(width=2),
            hovertemplate='<b>Frequency</b>: %{x:.1f} Hz<br>' +
                         '<b>ASD</b>: %{y:.2e} 1/√Hz<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{detector_name} Amplitude Spectral Density',
            xaxis_title='Frequency (Hz)',
            yaxis_title='ASD (1/√Hz)',
            xaxis_type='log',
            yaxis_type='log',
            height=400,
            template='plotly_white'
        )
        
        # Limit frequency range
        fig.update_xaxes(range=[np.log10(10), np.log10(2000)])
        
        return fig
    
    def plot_q_transform(self, time, strain, detector_name):
        """Create Q-transform plot (time-frequency representation)"""
        try:
            from gwpy.timeseries import TimeSeries
            
            # Convert to gwpy TimeSeries for Q-transform
            ts = TimeSeries(strain, sample_rate=4096, epoch=time[0])
            qgram = ts.q_transform(outseg=(time[0], time[-1]))
            
            # Create plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=qgram.value.T,
                x=qgram.times.value,
                y=qgram.frequencies.value,
                colorscale='Viridis',
                colorbar=dict(title="Normalized Energy"),
                hovertemplate='<b>Time</b>: %{x:.3f} s<br>' +
                             '<b>Frequency</b>: %{y:.1f} Hz<br>' +
                             '<b>Energy</b>: %{z:.2f}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'{detector_name} Q-Transform',
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                yaxis_type='log',
                height=400,
                template='plotly_white'
            )
            
            return fig
            
        except ImportError:
            # Fallback to regular spectrogram if gwpy not available
            return self.plot_spectrogram(time, strain, detector_name)
        except Exception as e:
            st.warning(f"Q-transform failed: {str(e)}. Showing spectrogram instead.")
            return self.plot_spectrogram(time, strain, detector_name)
    
    def plot_whitened_strain(self, original_strain, whitened_strain, time, detector_name):
        """Compare original and whitened strain data"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Original Strain', 'Whitened Strain'],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Original strain
        fig.add_trace(
            go.Scatter(
                x=time, y=original_strain,
                mode='lines', name='Original',
                line=dict(width=1, color=self.colors[0])
            ),
            row=1, col=1
        )
        
        # Whitened strain  
        fig.add_trace(
            go.Scatter(
                x=time, y=whitened_strain,
                mode='lines', name='Whitened',
                line=dict(width=1, color=self.colors[1])
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{detector_name} Strain Comparison',
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Strain", row=1, col=1)
        fig.update_yaxes(title_text="Whitened Strain", row=2, col=1)
        
        return fig
    
    def plot_skymap(self, skymap_data, projection="mollweide", coord_system="equatorial"):
        """Create sky map visualization with different projections"""
        if not SKYMAP_AVAILABLE or skymap_data is None:
            st.error("Sky mapping functionality not available")
            return go.Figure()
        
        try:
            map_data = skymap_data['map_data']
            nside = skymap_data['nside']
            
            if projection == "mollweide":
                return self._plot_mollweide_skymap(skymap_data, coord_system)
            elif projection == "lambert":
                return self._plot_lambert_skymap(skymap_data, coord_system)
            elif projection == "cartesian":
                return self._plot_cartesian_skymap(skymap_data, coord_system)
            else:
                return self._plot_mollweide_skymap(skymap_data, coord_system)
                
        except Exception as e:
            st.error(f"Error plotting sky map: {str(e)}")
            return go.Figure()
    
    def _plot_mollweide_skymap(self, skymap_data, coord_system="equatorial"):
        """Create Mollweide projection sky map"""
        map_data = skymap_data['map_data']
        nside = skymap_data['nside']
        
        # Convert to grid for plotting
        lon_grid, lat_grid, map_grid = self._skymap_to_grid(skymap_data, coord_system)
        
        fig = go.Figure()
        
        # Create the sky map using a heatmap with custom projection-like appearance
        fig.add_trace(go.Heatmap(
            z=map_grid,
            x=lon_grid[0],
            y=lat_grid[:, 0],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=self._get_colorbar_title(skymap_data['map_type']),
                titleside="right"
            ),
            hovertemplate='<b>Longitude</b>: %{x:.1f}°<br>' +
                         '<b>Latitude</b>: %{y:.1f}°<br>' +
                         '<b>Value</b>: %{z:.2e}<br>' +
                         '<extra></extra>'
        ))
        
        coord_label = "RA/Dec" if coord_system == "equatorial" else "Galactic l/b"
        
        fig.update_layout(
            title=f'{skymap_data["map_type"].upper()} Sky Map ({skymap_data["gwb_model"]} model) - Mollweide Projection',
            xaxis_title=f'Longitude ({coord_label}) [degrees]',
            yaxis_title=f'Latitude ({coord_label}) [degrees]',
            width=800,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _plot_lambert_skymap(self, skymap_data, coord_system="equatorial"):
        """Create Lambert azimuthal equal-area projection sky map"""
        map_data = skymap_data['map_data']
        nside = skymap_data['nside']
        
        # Convert to Lambert projection coordinates
        lon_grid, lat_grid, map_grid = self._skymap_to_grid(skymap_data, coord_system, projection="lambert")
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=map_grid,
            x=lon_grid[0],
            y=lat_grid[:, 0],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(
                title=self._get_colorbar_title(skymap_data['map_type']),
                titleside="right"
            )
        ))
        
        coord_label = "RA/Dec" if coord_system == "equatorial" else "Galactic l/b"
        
        fig.update_layout(
            title=f'{skymap_data["map_type"].upper()} Sky Map - Lambert Projection',
            xaxis_title=f'X ({coord_label})',
            yaxis_title=f'Y ({coord_label})',
            width=600,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def _plot_cartesian_skymap(self, skymap_data, coord_system="equatorial"):
        """Create Cartesian projection sky map"""
        lon_grid, lat_grid, map_grid = self._skymap_to_grid(skymap_data, coord_system)
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=map_grid,
            x=lon_grid[0],
            y=lat_grid[:, 0],
            colorscale='Cividis',
            showscale=True,
            colorbar=dict(
                title=self._get_colorbar_title(skymap_data['map_type']),
                titleside="right"
            )
        ))
        
        coord_label = "RA/Dec" if coord_system == "equatorial" else "Galactic l/b"
        
        fig.update_layout(
            title=f'{skymap_data["map_type"].upper()} Sky Map - Cartesian Projection',
            xaxis_title=f'Longitude ({coord_label}) [degrees]',
            yaxis_title=f'Latitude ({coord_label}) [degrees]',
            width=800,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _skymap_to_grid(self, skymap_data, coord_system="equatorial", projection="mollweide", resolution=100):
        """Convert HEALPix sky map to regular grid for plotting"""
        map_data = skymap_data['map_data']
        nside = skymap_data['nside']
        
        if coord_system == "equatorial":
            lon_range = (0, 360)
            lat_range = (-90, 90)
        else:  # galactic
            lon_range = (0, 360)
            lat_range = (-90, 90)
        
        # Create coordinate grids
        lon = np.linspace(lon_range[0], lon_range[1], resolution)
        lat = np.linspace(lat_range[0], lat_range[1], resolution//2)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Convert to HEALPix angles
        if coord_system == "equatorial":
            theta = (90 - lat_grid) * np.pi / 180
            phi = lon_grid * np.pi / 180
        else:
            theta = (90 - lat_grid) * np.pi / 180
            phi = lon_grid * np.pi / 180
        
        # Interpolate HEALPix data to grid
        try:
            map_grid = hp.get_interp_val(map_data, theta.flatten(), phi.flatten())
            map_grid = map_grid.reshape(theta.shape)
        except:
            # Fallback to nearest neighbor
            pix = hp.ang2pix(nside, theta.flatten(), phi.flatten())
            map_grid = map_data[pix].reshape(theta.shape)
        
        return lon_grid, lat_grid, map_grid
    
    def _get_colorbar_title(self, map_type):
        """Get appropriate colorbar title for map type"""
        if map_type == "snr":
            return "SNR"
        elif map_type == "upper_limit":
            return "Upper Limit [strain]"
        elif map_type == "detection_prob":
            return "Detection Probability"
        else:
            return "Value"
    
    def plot_interactive_skymap_controls(self, skymap_data):
        """Create interactive controls for sky map manipulation"""
        if not SKYMAP_AVAILABLE or skymap_data is None:
            return go.Figure()
        
        # This will be used with Streamlit widgets for interactivity
        # The actual interactive controls are handled in the main app
        return self.plot_skymap(skymap_data)
