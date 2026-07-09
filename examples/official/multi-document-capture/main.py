import sys
from PySide6.QtWidgets import QApplication
from app import DocumentScannerApp


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DocumentScannerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
