import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from utils.data_processing import DataProcessor
from utils.visualization import GWVisualizer

# Configure page
st.set_page_config(
    page_title="LIGO Gravitational Wave Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = GWVisualizer()

def main():
    st.title("🌊 LIGO Gravitational Wave Explorer")
    st.markdown("Explore real gravitational wave data from LIGO detectors")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Event selection
        st.subheader("Select Event")
        famous_events = {
            "GW150914": {"gps": 1126259462.4, "description": "First direct detection of gravitational waves"},
            "GW151226": {"gps": 1135136350.6, "description": "Second confirmed detection"},
            "GW170104": {"gps": 1167559936.6, "description": "Third LIGO detection"},
            "GW170814": {"gps": 1187008882.4, "description": "First three-detector observation"},
            "GW170817": {"gps": 1187529256.5, "description": "Neutron star merger with electromagnetic counterpart"},
            "GW190521": {"gps": 1242442967.4, "description": "Most massive black hole merger"},
            "Custom": {"gps": None, "description": "Enter custom GPS time"}
        }
        
        selected_event = st.selectbox(
            "Choose a gravitational wave event:",
            list(famous_events.keys()),
            format_func=lambda x: f"{x} - {famous_events[x]['description']}"
        )
        
        if selected_event == "Custom":
            gps_time = st.number_input(
                "Enter GPS time:",
                min_value=1000000000,
                max_value=2000000000,
                value=1126259462,
                help="GPS time of the event center"
            )
        else:
            gps_time = famous_events[selected_event]["gps"]
            st.info(f"GPS Time: {gps_time}")
        
        # Detector selection
        st.subheader("Detectors")
        detectors = st.multiselect(
            "Select detectors:",
            ["H1", "L1", "V1"],
            default=["H1", "L1"],
            help="H1: Hanford, L1: Livingston, V1: Virgo"
        )
        
        # Time window
        st.subheader("Time Window")
        duration = st.slider(
            "Duration around event (seconds):",
            min_value=4,
            max_value=64,
            value=32,
            step=4
        )
        
        # Signal processing options
        st.subheader("Signal Processing")
        apply_bandpass = st.checkbox("Apply bandpass filter", value=True)
        if apply_bandpass:
            low_freq = st.slider("Low frequency (Hz):", 10, 100, 35)
            high_freq = st.slider("High frequency (Hz):", 200, 2000, 350)
        
        apply_whitening = st.checkbox("Apply whitening", value=False)
        
        # Download button
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Downloading data from GWOSC..."):
                success = st.session_state.data_processor.load_data(
                    gps_time, duration, detectors, 
                    apply_bandpass, low_freq if apply_bandpass else None, 
                    high_freq if apply_bandpass else None, apply_whitening
                )
                if success:
                    st.success("Data loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load data. Please check your parameters.")
    
    # Main content area
    if st.session_state.data_processor.has_data():
        display_data_analysis()
    else:
        display_welcome_screen()

def display_welcome_screen():
    st.markdown("""
    ## Welcome to the LIGO Gravitational Wave Explorer! 🚀
    
    This app allows you to explore real gravitational wave data from the LIGO Scientific Collaboration.
    
    ### How to get started:
    1. **Select an Event**: Choose from famous gravitational wave detections or enter a custom GPS time
    2. **Choose Detectors**: Select which LIGO/Virgo detectors to analyze
    3. **Configure Analysis**: Set time window and signal processing options
    4. **Load Data**: Click the "Load Data" button to download real data from GWOSC
    
    ### About Gravitational Waves:
    Gravitational waves are ripples in the fabric of spacetime caused by accelerating masses. 
    LIGO (Laser Interferometer Gravitational-Wave Observatory) can detect these incredibly 
    small distortions - smaller than 1/10,000th the width of a proton!
    
    ### Famous Detections:
    - **GW150914**: The first direct detection of gravitational waves (September 14, 2015)
    - **GW170817**: A neutron star merger observed in both gravitational waves and light
    - **GW190521**: The most massive black hole merger detected so far
    
    Select an event from the sidebar and click "Load Data" to begin exploring! 📊
    """)
    
    # Educational content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 What is LIGO?")
        st.markdown("""
        LIGO consists of two identical L-shaped detectors:
        - **Hanford, Washington (H1)**
        - **Livingston, Louisiana (L1)**
        
        Each detector uses laser interferometry to measure tiny changes in the distance 
        between mirrors caused by passing gravitational waves.
        """)
    
    with col2:
        st.subheader("📡 Data Sources")
        st.markdown("""
        This app uses real data from:
        - **GWOSC**: Gravitational-Wave Open Science Center
        - **LIGO Scientific Collaboration**
        - **Virgo Collaboration**
        
        All data is publicly available and used for scientific research worldwide.
        """)

def display_data_analysis():
    data = st.session_state.data_processor.get_data()
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🎵 Spectrograms", "📊 Comparison", "📋 Data Info"])
    
    with tab1:
        st.subheader("Gravitational Wave Strain Data")
        
        # Plot time series for each detector
        for detector in data.keys():
            st.markdown(f"### {detector} Detector")
            fig = st.session_state.visualizer.plot_strain_timeseries(
                data[detector]['time'], 
                data[detector]['strain'], 
                detector
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Frequency Analysis")
        
        for detector in data.keys():
            st.markdown(f"### {detector} Spectrogram")
            fig = st.session_state.visualizer.plot_spectrogram(
                data[detector]['time'],
                data[detector]['strain'],
                detector
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if len(data) > 1:
            st.subheader("Multi-Detector Comparison")
            fig = st.session_state.visualizer.plot_multi_detector_comparison(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cross-correlation analysis
            if len(data) == 2:
                detectors = list(data.keys())
                st.subheader("Cross-Correlation Analysis")
                correlation = st.session_state.data_processor.calculate_cross_correlation(
                    data[detectors[0]]['strain'],
                    data[detectors[1]]['strain']
                )
                
                fig_corr = go.Figure()
                lags = np.arange(-len(correlation)//2, len(correlation)//2) * (1/4096)  # Assuming 4096 Hz
                fig_corr.add_trace(go.Scatter(
                    x=lags,
                    y=correlation,
                    mode='lines',
                    name='Cross-correlation'
                ))
                fig_corr.update_layout(
                    title=f"Cross-correlation: {detectors[0]} vs {detectors[1]}",
                    xaxis_title="Time lag (s)",
                    yaxis_title="Correlation"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Select multiple detectors to see comparison plots.")
    
    with tab4:
        st.subheader("Data Information")
        
        for detector in data.keys():
            with st.expander(f"{detector} Detector Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sampling Rate", f"{data[detector]['sample_rate']} Hz")
                    st.metric("Duration", f"{len(data[detector]['strain'])/data[detector]['sample_rate']:.1f} s")
                    st.metric("Data Points", f"{len(data[detector]['strain']):,}")
                
                with col2:
                    strain_data = data[detector]['strain']
                    st.metric("Mean Strain", f"{np.mean(strain_data):.2e}")
                    st.metric("RMS Strain", f"{np.sqrt(np.mean(strain_data**2)):.2e}")
                    st.metric("Peak Strain", f"{np.max(np.abs(strain_data)):.2e}")
        
        # Processing information
        processing_info = st.session_state.data_processor.get_processing_info()
        if processing_info:
            st.subheader("Signal Processing Applied")
            for key, value in processing_info.items():
                st.write(f"**{key}**: {value}")

if __name__ == "__main__":
    main()
