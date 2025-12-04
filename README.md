# Simple-Fraud
Contains the files to replicate the work I did for my senior capstone on fraud. 

# Installtion Guide
1. Go to the GitHub repository: `acrossler/Simple-Fraud`. This contains the files needed to replicate the work completed for my senior capstone on fraud.

2. Download the `CreditCards.bak` file from the repository.

3. Open SQL Server Management Studio.  
   If you need a server, download Microsoft SQL Server Express:  
   *Download Microsoft® SQL Server® 2022 Express from the Official Microsoft Download Center.*

4. Sign onto your server and create a database called `CreditCards`.

5. After this, right-click on **CreditCards**, then go to:  
   **Tasks → Policies → Backups**.  
   From there click **Add**, then click the **three dots (…)**.  
   In *Selected Path*, select all and paste into a notepad.  
   You should get a path similar to:  
   ```
   C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS2022\MSSQL\Backup
   ```

6. Go to that location on your machine and paste the downloaded file `CreditCards.bak` into the folder.

7. Then, right-click on **CreditCards**, go to **Tasks → Take Offline**.  
   After this, right-click on CreditCards again, then go to:  
   **Tasks → Policies → Restore → Database**.  
   From there, select **OK** and the database will be restored.  
   To confirm everything is working, go to **Tables** under CreditCards and right-click **Transaction** and **User**, selecting **Select Top 1000 Rows** for each.  
   - `Transaction` should return **1000 rows**  
   - `User` should return **236 rows**

8. Now it is time to connect the database to Python.  
   If you want to skip this step, download the CSV files `transaction3.csv` and `user3.csv` from GitHub and skip to **Step 11**.  
   Make sure to still run **Step 9**.

   If *not* skipping, then while the database is open, open the file `connectionToDatabase.py` in your Python environment (I use Spyder).  
   Ensure the server name matches your SQL Server instance.  
   Change the `SQLEXPRESS2022` part after:
   ```
   Server = localhost\\
   ```
   …to match your installation.

9. Run this line to install all required packages in your Python environment (e.g., Spyder console):
   ```
   pip install pandas geopy statsmodels numpy matplotlib scikit-learn tensorflow seaborn xgboost pyodbc
   ```

10. Then run `connectionToDatabase.py`.

11. Download and run `trainTestSplit.py`.  
    It contains all of the models and will show the results.
