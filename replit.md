# Overview

LIGO Gravitational Wave Explorer is a Streamlit-based interactive web application designed to explore and visualize real gravitational wave data from LIGO (Laser Interferometer Gravitational-Wave Observatory) detectors. The application provides an intuitive interface for analyzing famous gravitational wave events like GW150914 (the first direct detection) and other significant discoveries, allowing users to examine strain data, spectrograms, and various signal processing techniques applied to gravitational wave detection.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and scientific visualization
- **Layout**: Wide layout with expandable sidebar for configuration controls
- **State Management**: Streamlit session state for persistent data storage across user interactions
- **UI Components**: Sidebar-based parameter controls with main content area for visualizations

## Data Processing Architecture
- **Core Module**: `DataProcessor` class handles all gravitational wave data operations
- **Data Source**: GWOSC (Gravitational Wave Open Science Center) via gwpy library
- **Signal Processing**: Scipy-based filtering including bandpass filters and whitening techniques
- **Caching Strategy**: Built-in gwpy caching for downloaded data to improve performance

## Visualization Architecture
- **Primary Library**: Plotly for interactive scientific visualizations
- **Visualization Module**: `GWVisualizer` class encapsulates all plotting functionality
- **Chart Types**: Time series plots, spectrograms, and frequency domain analysis
- **Interactivity**: Hover tooltips, zoom capabilities, and responsive design

## Data Flow Pattern
- **Event Selection**: User selects from predefined famous events or custom GPS times
- **Data Fetching**: Real-time download from LIGO open data repositories
- **Processing Pipeline**: Configurable signal processing with bandpass filtering and whitening
- **Visualization**: Multi-detector strain data presentation with comparative analysis

## Error Handling Strategy
- **Graceful Degradation**: Handles missing gwpy dependency with user-friendly error messages
- **Data Validation**: GPS time and parameter validation before data requests
- **Network Resilience**: Caching mechanism for handling intermittent connectivity issues

# External Dependencies

## Scientific Computing Stack
- **NumPy**: Numerical computations and array operations
- **SciPy**: Signal processing functions for filtering and spectral analysis
- **Pandas**: Data manipulation and time series handling
- **Matplotlib**: Additional plotting capabilities and backend support

## Gravitational Wave Data Access
- **gwpy**: Primary interface to LIGO gravitational wave data
- **GWOSC**: Gravitational Wave Open Science Center data repository
- **LIGO Data Grid**: Real-time access to detector strain data

## Visualization and UI
- **Streamlit**: Web application framework and deployment platform
- **Plotly**: Interactive plotting library for scientific visualizations
- **Plotly Express**: Simplified plotting interface for rapid development

## Time and Date Handling
- **datetime**: GPS time conversion and event scheduling
- **timedelta**: Duration calculations for data windows

## Development Dependencies
- **warnings**: Error suppression for cleaner user experience during data processing operations