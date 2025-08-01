# 🎯 Gap Trading System

[![codecov](https://codecov.io/gh/yourusername/gappers-trader/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/gappers-trader)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-grade overnight gap trading system with live paper-trading support via Alpaca. Built for Python 3.12+ with comprehensive backtesting, risk management, and real-time execution capabilities.

## ✨ Features

### 🔍 **Signal Generation**
- **Survivorship-bias-free** gap detection using multiple data sources
- **Advanced ranking** with sector diversification and technical indicators
- **Real-time scanning** with customizable gap thresholds

### 📊 **Backtesting Engine**
- **Vectorized simulation** using vectorbt for 20+ years of data in <60 seconds
- **Realistic execution** with slippage, commissions, and SEC Rule 201 compliance
- **Parameter optimization** with grid search and sensitivity analysis

### 📈 **Analytics & Visualization**
- **Interactive Streamlit dashboard** with real-time performance metrics
- **Comprehensive risk analysis** including VaR, CVaR, and drawdown metrics
- **Professional reports** with detailed trade analysis and benchmarking

### ⚡ **Live Trading**
- **Alpaca Paper Trading** integration with bracket orders
- **Scheduled execution** with pre-market scanning and automatic position management
- **Risk controls** with position sizing, sector limits, and time-based exits

### 🛠️ **Production Ready**
- **Multi-source data feeds** (yfinance, IEX Cloud, Polygon.io)
- **Docker containerization** with multi-stage builds
- **CI/CD pipeline** with comprehensive testing and security scanning
- **Monitoring & observability** with structured logging and health checks

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 16GB RAM recommended for full backtests
- Ubuntu 24.04 or compatible Linux distribution

### Installation

#### Option 1: Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/gappers-trader.git
cd gappers-trader

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

#### Option 2: Using Docker

```bash
# Clone and run with Docker Compose
git clone https://github.com/yourusername/gappers-trader.git
cd gappers-trader

# Start the Streamlit dashboard
docker-compose up gap-trader

# Access dashboard at http://localhost:8501
```

#### Option 3: Using pip

```bash
pip install gappers-trader

# Or install from source
pip install git+https://github.com/yourusername/gappers-trader.git
```

### Basic Usage

#### 1. **Streamlit Dashboard**

```bash
# Start the interactive dashboard
streamlit run app.py

# Navigate to http://localhost:8501
```

#### 2. **Command Line Interface**

```bash
# Run a backtest
gappers backtest --start-date 2020-01-01 --end-date 2023-12-31 --profit-target 0.05

# Scan for today's gaps
gappers scan --min-gap 0.02 --top-k 10

# Parameter sweep optimization  
gappers sweep --profit-target 0.03 0.05 0.07 --stop-loss 0.01 0.02 --output sweep_results.csv

# Live paper trading (requires Alpaca credentials)
gappers live --dry-run
```

#### 3. **Python API**

```python
from datetime import datetime
from gappers import DataFeed, SignalGenerator, Backtester, GapParams

# Initialize components
data_feed = DataFeed()
signal_gen = SignalGenerator(data_feed)
backtester = Backtester(data_feed, signal_gen)

# Configure strategy parameters
params = GapParams(
    profit_target=0.05,  # 5% profit target
    stop_loss=0.02,      # 2% stop loss
    top_k=10,            # Trade top 10 gaps
    position_size=10000, # $10k per position
)

# Run backtest
results = backtester.run_backtest(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    params=params
)

print(f"Total trades: {len(results['trades'])}")
print(f"Total return: {results['portfolio_values']['value'].iloc[-1] / 100000 - 1:.2%}")
```

## 📋 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Data Configuration
DATA_PATH=./data
LOG_LEVEL=INFO
CACHE_EXPIRY_HOURS=24

# Alpaca Paper Trading (Required for live trading)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional Premium Data Feeds
IEX_CLOUD_API_KEY=your_iex_cloud_key
POLYGON_API_KEY=your_polygon_key

# Trading Configuration
DEFAULT_POSITION_SIZE=10000
DEFAULT_COMMISSION=0.005
DEFAULT_SLIPPAGE_BPS=10
```

### Strategy Parameters

Key parameters can be configured via the CLI, API, or Streamlit interface:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profit_target` | 0.05 | Profit target as decimal (5%) |
| `stop_loss` | 0.02 | Stop loss as decimal (2%) |
| `top_k` | 10 | Number of top gaps to trade |
| `min_gap_pct` | 0.02 | Minimum gap size (2%) |
| `max_gap_pct` | 0.30 | Maximum gap size (30%) |
| `max_hold_time_hours` | 6 | Maximum hold time |
| `sector_diversification` | True | Enable sector limits |
| `max_per_sector` | 3 | Max positions per sector |

## 📊 Performance Metrics

The system calculates comprehensive performance metrics:

### **Return Metrics**
- Total Return, CAGR, Sharpe Ratio, Sortino Ratio
- Win Rate, Average Winner/Loser, Profit Factor

### **Risk Metrics**  
- Maximum Drawdown, Value at Risk (95%, 99%)
- Conditional VaR, Beta, Tracking Error

### **Trade Analysis**
- Distribution by exit reason, hold time, sector
- Temporal patterns (monthly, daily, hourly performance)
- Gap size vs. return correlation analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Signal Engine  │    │ Execution Engine│
│                 │    │                 │    │                 │
│ • yfinance      │───▶│ • Gap Detection │───▶│ • Backtesting   │
│ • IEX Cloud     │    │ • Ranking       │    │ • Live Trading  │
│ • Polygon.io    │    │ • Filtering     │    │ • Risk Mgmt     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Universe Builder│    │   Analytics     │    │  Interfaces     │
│                 │    │                 │    │                 │
│ • Liquidity     │    │ • Performance   │    │ • Streamlit UI  │
│ • Survivorship  │    │ • Risk Analysis │    │ • CLI           │
│ • Filtering     │    │ • Reporting     │    │ • Python API    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Core Components**

- **DataFeed**: Multi-source data ingestion with intelligent caching
- **UniverseBuilder**: Survivorship-bias-free symbol universe construction
- **SignalGenerator**: Gap detection, ranking, and technical analysis
- **Backtester**: Vectorized simulation engine with realistic execution
- **LiveTrader**: Real-time execution via Alpaca with risk management
- **PerformanceAnalyzer**: Comprehensive analytics and visualization

## 🧪 Testing

The system includes comprehensive tests with 80%+ coverage:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=gappers --cov-report=html

# Run specific test categories  
poetry run pytest tests/test_backtest.py -v
```

### **Test Categories**
- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end workflow testing  
- **Performance Tests**: Backtesting speed and memory usage
- **Live Trading Tests**: Mock execution and risk management

## 🐳 Docker Deployment

### **Development**

```bash
# Build and run development container
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run CLI commands
docker-compose run --rm gap-trader-cli scan --help
```

### **Production**

```bash
# Production deployment with monitoring
docker-compose --profile monitoring up -d

# Scale for high throughput
docker-compose up --scale gap-trader=3
```

### **Services Available**

- `gap-trader`: Streamlit dashboard (port 8501)
- `gap-trader-cli`: Command-line interface
- `gap-trader-live`: Live trading service
- `redis`: Caching layer (optional)
- `postgres`: Trade history storage (optional)
- `prometheus`: Metrics collection (optional)
- `grafana`: Monitoring dashboard (optional)

## 📈 Usage Examples

### **Basic Backtesting**

```python
from gappers import Backtester, GapParams
from datetime import datetime

# 5-year backtest with optimization
backtester = Backtester()
params = GapParams(profit_target=0.06, stop_loss=0.025)

results = backtester.run_backtest(
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 1, 1), 
    params=params
)

# Analyze results
analyzer = PerformanceAnalyzer()
analysis = analyzer.analyze_backtest_results(results)
report = analyzer.generate_performance_report(analysis)
print(report)
```

### **Live Paper Trading**

```python
from gappers import LiveTrader, GapParams

# Initialize live trader
trader = LiveTrader(dry_run=False)  # Set to False for actual paper trading
params = GapParams(top_k=5, profit_target=0.04)

# Run single scan
opportunities = trader.run_single_scan(params)
print(f"Found {opportunities['gaps_found']} opportunities")

# Start scheduled trading (runs until stopped)
trader.start_live_trading(params)
```

### **Advanced Analysis**

```python
from gappers import PerformanceAnalyzer
import matplotlib.pyplot as plt

analyzer = PerformanceAnalyzer()

# Create detailed trade analysis plots
figures = analyzer.create_trade_analysis_plots(
    trades=results['trades'],
    save_dir='./analysis_plots'
)

# Generate interactive dashboard
dashboard = analyzer.create_performance_dashboard(analysis)
dashboard.write_html('performance_dashboard.html')
```

## 🛡️ Security & Compliance

### **Security Features**
- **No hardcoded secrets** - all credentials via environment variables
- **Input validation** on all user inputs and API responses
- **Rate limiting** and request throttling for external APIs
- **Audit logging** of all trades and system events

### **Compliance**
- **SEC Rule 201** compliance for short sale restrictions
- **Position limits** and risk management controls
- **Trade reporting** with full audit trail
- **Data retention** policies for regulatory requirements

### **Risk Management**
- **Position sizing** with maximum exposure limits
- **Sector diversification** to limit concentration risk
- **Time-based exits** to limit overnight exposure
- **Circuit breakers** for unusual market conditions

## 🔧 Development

### **Setting up Development Environment**

```bash
# Clone and setup
git clone https://github.com/yourusername/gappers-trader.git
cd gappers-trader

# Install with development dependencies
poetry install --with dev

# Setup pre-commit hooks
poetry run pre-commit install

# Run development server
poetry run streamlit run app.py --server.runOnSave true
```

### **Code Quality**

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Fast Python linter  
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning
- **Pre-commit**: Git hooks for code quality

```bash
# Run all quality checks
poetry run pre-commit run --all-files

# Individual tools
poetry run black gappers/
poetry run ruff check gappers/
poetry run mypy gappers/
poetry run bandit -r gappers/
```

## 📚 Documentation

- **[API Reference](docs/api.md)**: Complete API documentation
- **[Strategy Guide](docs/strategy.md)**: Gap trading strategy explanation
- **[Configuration](docs/configuration.md)**: Detailed configuration options
- **[Deployment](docs/deployment.md)**: Production deployment guide
- **[Contributing](CONTRIBUTING.md)**: Development and contribution guidelines

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### **Development Workflow**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss and is not suitable for all investors. Always consult with a qualified financial advisor before making investment decisions.

The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## 🙏 Acknowledgments

- **[vectorbt](https://vectorbt.dev/)**: High-performance backtesting framework
- **[Streamlit](https://streamlit.io/)**: Interactive web application framework  
- **[Alpaca](https://alpaca.markets/)**: Commission-free trading API
- **[yfinance](https://github.com/ranaroussi/yfinance)**: Yahoo Finance data access

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/gappers-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gappers-trader/discussions)
- **Documentation**: [Project Wiki](https://github.com/yourusername/gappers-trader/wiki)

---

<div align="center">

**Built with ❤️ for the trading community**

[⭐ Star this repo](https://github.com/yourusername/gappers-trader) • [🐛 Report Bug](https://github.com/yourusername/gappers-trader/issues) • [💡 Request Feature](https://github.com/yourusername/gappers-trader/issues)

</div>
