# Simple-Fraud
Contains the files to replicate the work I did for my senior capstone on fraud. 

# Installtion Guide
1. Go to the GitHub repository: `acrossler/Simple-Fraud`. This contains the files needed to replicate the work I did for my senior capstone on fraud.

2. Download the `CreditCardsScript.sql` file from GitHub.

3. Open SQL Server Management Studio.  
   If you need a server, download Microsoft SQL Server Express:  
   *Download Microsoft® SQL Server® 2022 Express from the Official Microsoft Download Center.*

4. Sign onto your server and create a database called `CreditCards`.

5. Open `CreditCardsScript.sql` and run it.  
   To confirm everything loaded correctly:
   - Expand the **CreditCards** database → **Tables**
   - Right-click **Transaction** → *Select Top 1000 Rows*  
   - Right-click **User** → *Select Top 1000 Rows*  
   You should see:
   - **Transaction**: 1000 rows  
   - **User**: 236 rows

6. Now it is time to connect the database to Python.  
   If you do not have Python installed, download Anaconda and use Spyder:  
   https://www.anaconda.com/download  
   
   **If you want to skip database connection**, download `transaction3.csv` and `user3.csv` from GitHub and skip to **Step 9**.  
   Make sure you still run **Step 7**.

   If *not skipping*, then:
   - Open `connectionToDatabase.py` in your Python environment (I use Spyder).  
   - Ensure the server name matches your SQL Server instance.  
     Change the `SQLEXPRESS2022` part after:
     ```
     Server = localhost\\
     ```
     to match your setup.

7. Install all required packages by running this line in your Python console/terminal:
   ```
   pip install pandas geopy statsmodels numpy matplotlib scikit-learn tensorflow seaborn xgboost pyodbc
   ```

8. Run `connectionToDatabase.py`.

9. Download and run `trainTestSplit.py`.  
   It includes all of the models and displays the results.  
   **Make sure the CSV files are in the same folder as `trainTestSplit.py`** or you will get an error.  
   You can change the file paths inside `trainTestSplit.py` if needed.

   In the confusion matrix:
   - Rows = actual classes  
   - Columns = predicted classes  
   - First row/column = *not fraud*  
   - Second row/column = *fraud*  
   The value printed below the matrix is the total accuracy.
