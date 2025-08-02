# 🎯 GapRunner - Production Gap Trading System

A production-ready overnight gap trading system with real-time data collection, advanced analytics, and comprehensive backtesting capabilities.

## 🚀 Features

### 📊 **Two-Tier Architecture**
- **Tier 1**: Data Collection & Storage - Robust data pipeline with progress tracking
- **Tier 2**: Analysis & Trading - Streamlit dashboard for strategy execution

### 🔥 **Production-Ready Components**
- ✅ **Multi-source data providers** (Yahoo Finance, Polygon, Tiingo, Alpaca)
- ✅ **Robust error handling** with comprehensive logging
- ✅ **Production CLI tools** with rich progress bars
- ✅ **Partitioned Parquet storage** for efficient data access
- ✅ **Real-time gap detection** with technical indicators
- ✅ **Advanced portfolio simulation** engine
- ✅ **YAML configuration management** with validation
- ✅ **Security best practices** implemented throughout

### 📈 **Advanced Gap Analysis**
- Real-time gap calculation and ranking
- Technical indicator integration
- Historical pattern analysis
- Sector diversification support
- Risk management controls

### 🖥️ **Interactive Dashboard**
- Production Streamlit interface
- Real-time data visualization
- Configuration management UI
- Portfolio performance tracking
- Data validation tools

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER 1: DATA LAYER                      │
├─────────────────────────────────────────────────────────────┤
│  Data Collection → Storage → Validation → Management       │
│  • Multi-source APIs     • Parquet files    • CLI tools   │
│  • Progress tracking     • Partitioning     • Monitoring   │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   TIER 2: ANALYSIS LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  Gap Engine → Portfolio Sim → Dashboard → Strategy Exec    │
│  • Real-time gaps    • P&L tracking    • Web interface     │
│  • Ranking system    • Risk metrics    • Configuration     │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- 8GB+ RAM recommended
- 10GB+ disk space for data storage

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd GapRunner/gappers_trader

# Install dependencies
pip install -r requirements.txt

# Or with Poetry
poetry install

# Install with premium data sources
poetry install --extras premium
```

### Environment Setup

Create a `.env` file for API keys (optional):

```bash
# Premium data sources (optional)
POLYGON_API_KEY=your_polygon_key
TIINGO_API_KEY=your_tiingo_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
```

## 🚦 Quick Start Guide

### 1. Data Collection (Tier 1)

```bash
# Collect last 30 days of data
python cli_collect.py --days 30

# Collect specific date range
python cli_collect.py --start-end 2024-01-01 2024-01-31

# Validate data integrity
python cli_collect.py --days 7 --validate
```

### 2. Gap Analysis (CLI)

```bash
# Analyze yesterday's gaps
python cli_gaps.py --yesterday

# Analyze specific date
python cli_gaps.py --date 2024-01-15

# Export to CSV
python cli_gaps.py --date 2024-01-15 --export gaps.csv
```

### 3. Dashboard (Tier 2)

```bash
# Launch Streamlit dashboard
streamlit run app_new.py
```

Navigate to `http://localhost:8501` to access the dashboard.

## 📁 Project Structure

```
gappers_trader/
├── gappers/                    # Core system modules
│   ├── config_new.py          # YAML configuration management
│   ├── data_collector.py      # Production data collection
│   ├── data_manager.py        # Data storage & retrieval
│   ├── data_providers.py      # Multi-source data providers
│   ├── gap_engine.py          # Gap calculation & ranking
│   ├── universe.py            # Stock universe management
│   └── ...
├── app_new.py                 # Production Streamlit dashboard
├── cli_collect.py             # CLI data collection tool
├── cli_gaps.py                # CLI gap analysis tool
├── config.yaml                # System configuration
├── requirements.txt           # Python dependencies
└── README_NEW.md              # This file
```

## ⚙️ Configuration

The system uses YAML configuration files for all settings:

```yaml
# config.yaml
data_sources:
  primary: "yfinance"
  fallback: ["yfinance"]

data_collection:
  universe_size: 3000
  min_dollar_volume: 1000000
  frequency_minutes: 30

strategy:
  top_k: 10
  min_gap_pct: 0.02
  max_gap_pct: 0.30
  profit_target_pct: 0.10
  
# ... more configuration options
```

## 📊 Dashboard Features

### 🏠 Dashboard
- System overview and key metrics
- Quick gap analysis
- Real-time status monitoring

### 📊 Data Collection
- Interactive data collection interface
- Progress tracking with rich progress bars
- Data validation and integrity checks
- Storage statistics and management

### 🔍 Gap Analysis
- Real-time gap detection and ranking
- Historical pattern analysis
- Technical indicator integration
- Interactive charts and visualizations

### 💼 Portfolio Simulation
- Advanced backtesting engine
- P&L tracking with costs
- Risk metrics and drawdown analysis
- Strategy parameter optimization

### ⚙️ Configuration
- Web-based configuration management
- Parameter validation
- Multiple data source support
- Security settings

## 🔧 CLI Tools

### Data Collection CLI

```bash
# Basic usage
python cli_collect.py --days 30

# Advanced options
python cli_collect.py \
  --start-end 2024-01-01 2024-01-31 \
  --config custom_config.yaml \
  --validate \
  --force
```

### Gap Analysis CLI

```bash
# Basic analysis
python cli_gaps.py --yesterday

# Advanced analysis
python cli_gaps.py \
  --date 2024-01-15 \
  --direction up \
  --limit 10 \
  --export results.csv
```

## 🛡️ Security & Production Features

### Security
- ✅ API key encryption and secure storage
- ✅ Rate limiting and request throttling
- ✅ Input validation and sanitization
- ✅ Comprehensive audit logging
- ✅ Error handling without data exposure

### Production Readiness
- ✅ Robust error handling and recovery
- ✅ Comprehensive logging system
- ✅ Data validation and integrity checks
- ✅ Performance monitoring
- ✅ Scalable architecture
- ✅ Memory-efficient data processing
- ✅ Concurrent data collection
- ✅ Progress tracking and status reporting

### Performance
- ✅ Partitioned Parquet storage for fast queries
- ✅ Caching system for API responses
- ✅ Concurrent data processing
- ✅ Memory-efficient operations
- ✅ Optimized data structures

## 📈 Strategy Implementation

The system implements the momentum-gap strategy as specified:

### Core Strategy Rules
- **Entry**: Top K gap-up stocks at market open
- **Take Profit**: +10% from entry (configurable)
- **Trailing Stop**: 2% from session high (configurable)
- **Hard Stop**: -4% from entry (configurable)
- **Time Stop**: 15:55 ET exit (configurable)

### Risk Management
- Position sizing: $1,000 per position (configurable)
- Maximum positions: 10 (configurable)
- Sector diversification limits
- Portfolio risk limits (20% max exposure)

### Cost Model
- Commission: $0.005 per share
- Slippage: 10 basis points
- All costs included in P&L calculations

## 🧪 Testing & Validation

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=gappers tests/

# Lint code
ruff check gappers/
black --check gappers/

# Type checking
mypy gappers/
```

## 🚀 Deployment

### Docker Deployment (Coming Soon)
```bash
# Build container
docker build -t gap-trader .

# Run with data persistence
docker run -v ./data:/app/data -p 8501:8501 gap-trader
```

### Environment Variables
```bash
# Data sources
POLYGON_API_KEY=your_key
TIINGO_API_KEY=your_key

# System settings
DATA_PATH=./data
LOG_LEVEL=INFO
```

## 📚 API Reference

### Core Classes

#### `Config`
```python
from gappers.config_new import Config

config = Config.load("config.yaml")
issues = config.validate()  # Returns validation issues
```

#### `DataCollector`
```python
from gappers.data_collector import DataCollector

collector = DataCollector(config)
success = collector.collect_full_dataset(start_date, end_date)
```

#### `GapEngine`
```python
from gappers.gap_engine import GapEngine

engine = GapEngine(config)
gaps_df = engine.calculate_daily_gaps(date)
top_gaps = engine.get_top_gaps(date, direction="up", limit=10)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss and is not suitable for all investors. Please consult with a qualified financial advisor before making any investment decisions.

## 🆘 Support

- 📧 Email: support@gaprunner.com
- 🐛 Issues: [GitHub Issues](link-to-issues)
- 📖 Documentation: [Full Documentation](link-to-docs)

---

**Built with ❤️ for the trading community**