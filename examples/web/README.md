# Python Flask Web Document Scanner
This project demonstrates how to build a web-based document scanner using the **Dynamsoft Document Normalizer SDK** and **Flask**. The application leverages a connected camera to capture documents, processes them on the server-side, and presents the results in the web browser.

## Installation
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Prerequisites
- Obtain a [30-day free trial license](https://www.dynamsoft.com/customer/license/trialLicense/?product=ddn) for the Dynamsoft Document Normalizer SDK.

## How to Run 
1. **Set the License Key**: Update the license key in `document.py`:

    ```python
    docscanner.initLicense("LICENSE-KEY")
    ```

2. **Connect a Camera**: Ensure your camera is properly connected to your computer.
3. **Start the Application**: Run the Flask server and open the application in your web browser:

    ```bash
    python server.py
    ```
4. **Access the Application**: Visit `http://127.0.0.1:5000` in your web browser to use the document scanner.

    ![Python Flask web document scanner](https://www.dynamsoft.com/codepool/img/2024/08/python-flask-web-document-scanner.png)


