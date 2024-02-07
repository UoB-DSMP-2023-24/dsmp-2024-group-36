import pandas as pd

df = pd.read_csv('fake_transactional_data_24.csv')

df['not_happened_yet_date'] = pd.to_datetime(df['not_happened_yet_date'], format='%d/%m/%Y')
df['monopoly_money_amount'] = df['monopoly_money_amount'].astype(float)

def categorize_shop(row):
    if 'COFFEE_SHOP' in row:
        return 'Coffee Shop'
    elif 'CINEMA' in row:
        return 'Cinema'
    elif 'RESTAURANT' in row or 'CAFE' in row:
        return 'Restaurant'
    else:
        return 'Other'

df['Shop_Type'] = df['to_randomly_generated_account'].apply(categorize_shop)

def categorize_spending(amount):
    if amount < 3:
        return 'Low Spending'
    elif amount < 5:
        return 'Medium Spending'
    else:
        return 'High Spending'

df['Spending_Type'] = df['monopoly_money_amount'].apply(categorize_spending)

user_spending = df.groupby('from_totally_fake_account')['monopoly_money_amount'].agg(['sum', 'mean']).reset_index()
user_spending['Spending_Level'] = user_spending['mean'].apply(lambda x: 'Low' if x < 3 else ('Medium' if x < 5 else 'High'))

print(df)
print(user_spending)
