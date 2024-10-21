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

To make your dashboard accessible over the internet, you can use Ngrok. Follow these steps to set up and use Ngrok:

1. **Download Ngrok**:
   - Visit the [Ngrok download page](https://ngrok.com/download)
   - Download the version appropriate for your operating system

2. **Place ngrok.exe in the Project Folder**:
   - Extract the downloaded file
   - Move `ngrok.exe` to the root directory of your Income Analysis Dashboard project

3. **Set Up Ngrok Authentication**:
   - Sign up for a free Ngrok account at [https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup)
   - After signing in, go to the [Auth Token section](https://dashboard.ngrok.com/get-started/your-authtoken)
   - Copy your auth token
   - Open a command prompt or terminal in your project directory and run:
     ```bash
     ngrok config add-authtoken YOUR_AUTH_TOKEN
     ```
   Replace `YOUR_AUTH_TOKEN` with the token you copied.

4. **Start the Dash App**:
   ```bash
   python app.py
   ```

5. **Run Ngrok**:
   - Open a new command prompt or terminal window
   - Navigate to your project directory
   - Run the following command:
     ```bash
     ngrok http 8050
     ```
   - Ngrok will display a forwarding URL (e.g., `https://abc123def456.ngrok.io`)

6. **Access Your Dashboard**:
   - Share the Ngrok URL with others to give them access to your dashboard
   - The dashboard will be accessible as long as your local server and Ngrok are running

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

- Email: meighanba@vcu.edu
- GitHub: [@BradyMeighan](https://github.com/BradyMeighan)
- LinkedIn: [Brady Meighan](https://www.linkedin.com/in/BradyMeighan)

Happy Analyzing! 📈🔍
