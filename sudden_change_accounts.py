
import pandas as pd

# Load the dataset
data = pd.read_csv('fake_transactional_data_24.csv')
data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'], format='%d/%m/%Y')

# Summarize transactions by account
account_summary = data.groupby('from_totally_fake_account').agg(
    min_transaction_amount=('monopoly_money_amount', 'min'),
    max_transaction_amount=('monopoly_money_amount', 'max'),
).reset_index()

# Define threshold
range_threshold = account_summary['max_transaction_amount'].median() - account_summary['min_transaction_amount'].median()

# Filter accounts
sudden_change_accounts = account_summary[
    (account_summary['max_transaction_amount'] - account_summary['min_transaction_amount']) > range_threshold
]

# Save the results
sudden_change_accounts.to_csv('sudden_change_accounts.txt', index=False)
