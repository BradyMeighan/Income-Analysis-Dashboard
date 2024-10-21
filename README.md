# Income Analysis Dashboard

## Overview

The Income Analysis Dashboard is a powerful, interactive web application built with Python and Dash. It enables users to analyze income data, make predictions, and gain insights through visualizations of various demographic factors influencing income levels.

![Dashboard Preview](preview.png)

## 🚀 Features

- **Interactive Data Upload**: Upload your datasets or generate fake data for testing
- **Machine Learning Models**: Access multiple trained models for income prediction
- **Fairness Analysis**: Evaluate model fairness across demographic groups
- **Comprehensive Visualizations**: Gain insights through various chart types
- **Public Access**: Share your dashboard using Ngrok for remote access

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/income-analysis-dashboard.git
   cd income-analysis-dashboard
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the App

### Locally
```bash
python app.py
```
Access the dashboard at `http://localhost:8050`

### Publicly with Ngrok
1. Start the Dash app:
   ```bash
   python app.py
   ```
2. In a new terminal, run:
   ```bash
   ngrok.exe http 8050
   ```
3. Share the provided Ngrok URL for public access

## 📊 Usage

1. **Data Input**: Upload your data or generate fake data using `fakedata.py`
2. **Predictions**: Enter features or upload a dataset for batch predictions
3. **Analysis**: Explore visualizations and fairness metrics
4. **Export**: Download prediction results and analysis reports

## 📁 Project Structure

```
income-analysis-dashboard/
├── app.py
├── fakedata.py
├── requirements.txt
├── assets/
├── callbacks/
├── data/
├── fairness/
├── layout/
├── models/
├── tests/
└── utils/
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request with your features or fixes.

## 📄 License

This project is licensed under the MIT License.

## 📞 Contact

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://www.linkedin.com/in/yourname)

Happy Analyzing! 📈🔍
