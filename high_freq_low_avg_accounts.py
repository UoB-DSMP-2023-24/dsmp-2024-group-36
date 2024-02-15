
import pandas as pd

# Load the dataset
data = pd.read_csv('fake_transactional_data_24.csv')
data['not_happened_yet_date'] = pd.to_datetime(data['not_happened_yet_date'], format='%d/%m/%Y')

# Summarize transactions by account
account_summary = data.groupby('from_totally_fake_account').agg(
    total_transactions=('monopoly_money_amount', 'count'),
    average_transaction_amount=('monopoly_money_amount', 'mean'),
).reset_index()

# Define threshold
median_transactions = account_summary['total_transactions'].median()
median_avg_amount = account_summary['average_transaction_amount'].median()

# Filter accounts
high_freq_low_avg_accounts = account_summary[
    (account_summary['total_transactions'] > median_transactions) &
    (account_summary['average_transaction_amount'] < median_avg_amount)
]

# Save the results
high_freq_low_avg_accounts.to_csv('high_freq_low_avg_accounts.txt', index=False)
