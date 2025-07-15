print("Testing imports...")
try:
    import yfinance
    print("✅ yfinance OK")
except:
    print("❌ yfinance missing - run: pip install yfinance")

try:
    import pandas
    print("✅ pandas OK")
except:
    print("❌ pandas missing - run: pip install pandas")

try:
    import sklearn
    print("✅ sklearn OK")
except:
    print("❌ sklearn missing - run: pip install scikit-learn")

print("Import test complete!")
