# CLI Paper Trading System

A command-line interface for paper trading using the Gap Runner project's configurations and gap detection engine.

## Features

- **Configuration-based trading**: Use saved configurations from Portfolio Simulation
- **Real-time gap detection**: Leverage the project's gap detection engine
- **Alpaca API integration**: Execute paper trades through Alpaca Markets
- **Dry-run mode**: Test strategies without placing actual orders
- **Continuous trading**: Run automated trading sessions with scheduling
- **Rich CLI interface**: Beautiful terminal output with progress bars and tables

## Prerequisites

1. **Virtual Environment**: Activate the project's virtual environment:
   ```bash
   source ../venv/bin/activate
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install alpaca-trade-api schedule
   ```

3. **API Credentials**: Set up your Alpaca API credentials in `.env`:
   ```
   ALPACA_API_KEY=your_api_key_here
   ALPACA_API_SECRET=your_api_secret_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
   ```

4. **Saved Configurations**: Create trading configurations using the Streamlit app's Portfolio Simulation feature.

## Usage

### Test System

Test all dependencies and configurations:

```bash
python cli_paper_trading.py test
```

### List Available Configurations

View all saved trading configurations:

```bash
python cli_paper_trading.py list-configs
```

### Run Single Gap Scan

Execute a one-time gap scan and trading session:

```bash
# Dry run mode (no actual orders)
python cli_paper_trading.py scan <config_name> --dry-run

# Live trading mode
python cli_paper_trading.py scan <config_name>
```

### Continuous Trading

Run automated trading with periodic gap scans:

```bash
# Run with 5-minute intervals (default)
python cli_paper_trading.py run <config_name>

# Run with custom interval (in seconds)
python cli_paper_trading.py run <config_name> --scan-interval 600

# Run in dry-run mode
python cli_paper_trading.py run <config_name> --dry-run
```

### Check Account Status

View current account balance and positions:

```bash
python cli_paper_trading.py status <config_name>
```

## Example Workflow

1. **Create Configuration**: Use the Streamlit app to create and save a paper trading configuration.

2. **Test Setup**:
   ```bash
   python cli_paper_trading.py test
   python cli_paper_trading.py list-configs
   ```

3. **Dry Run Test**:
   ```bash
   python cli_paper_trading.py scan my-config --dry-run
   ```

4. **Live Trading**:
   ```bash
   python cli_paper_trading.py run my-config --scan-interval 300
   ```

## Configuration Parameters

The CLI uses the same configuration system as the Streamlit app, including:

- **Strategy Settings**: Entry/exit rules, profit targets, stop losses
- **Risk Management**: Position sizing, sector limits, exposure limits
- **Trading Hours**: Market hours, after-hours trading settings
- **Automation**: Scan intervals, trading schedules

## Logging

- **Console Output**: Real-time status updates and trading actions
- **Log File**: Detailed logging saved to `paper_trading.log`
- **Rich Interface**: Progress bars, tables, and colored status messages

## Safety Features

- **Dry Run Mode**: Test strategies without placing actual orders
- **Trading Hours Validation**: Respects market hours and configuration settings
- **Position Limits**: Enforces risk management rules from configurations
- **Error Handling**: Graceful handling of API failures and market closures
- **Graceful Shutdown**: Responds to Ctrl+C and system signals

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated and dependencies installed
2. **API Connection**: Verify Alpaca credentials in `.env` file
3. **No Configurations**: Create configurations using the Streamlit app first
4. **Market Hours**: Some features only work during trading hours

### Debug Mode

Run with Python's verbose logging:

```bash
python -v cli_paper_trading.py test
```

### Log Analysis

Check the log file for detailed error information:

```bash
tail -f paper_trading.log
```

## Integration

The CLI integrates seamlessly with the main Gap Runner project:

- Uses same configuration format as Streamlit app
- Leverages existing gap detection engine
- Shares universe and data management systems
- Compatible with saved simulation results