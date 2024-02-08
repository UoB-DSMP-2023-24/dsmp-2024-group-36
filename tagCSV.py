import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
csv_file_path = 'fake_transactional_data_24.csv'

df = pd.read_csv(csv_file_path)

df_filtered = df[df['to_randomly_generated_account'].str.contains(r'\D', regex=True)]
unique_shops = df_filtered['to_randomly_generated_account'].unique()

# print(f"{len(unique_shops)} shops in list")
# print("name list: ")
# for shop in unique_shops:
#     print(shop)
import pandas as pd

shop_tags_and_levels = {
    "CINEMA": ("Cinema", "Medium"),
    "HIPSTER_COFFEE_SHOP": ("Coffee Shop", "Medium"),
    "TOTALLY_A_REAL_COFFEE_SHOP": ("Coffee Shop", "Medium"),
    "COFFEE_SHOP": ("Coffee Shop", "Low"),
    "CAFE": ("Cafe", "Low"),
    "A_CAFE": ("Cafe", "Low"),
    "LOCAL_RESTAURANT": ("Restaurant", "Medium"),
    "A_LOCAL_COFFEE_SHOP": ("Coffee Shop", "Low"),
    "GOURMET_COFFEE_SHOP": ("High-End Coffee Shop", "High"),
    "LOCAL_WATERING_HOLE": ("Bar", "Medium"),
    "SANDWICH_SHOP": ("Fast Food Shop", "Low"),
    "TOY_SHOP": ("Toy Shop", "Low"),
    "PRETENTIOUS_COFFEE_SHOP": ("High-End Coffee Shop", "High"),
    "BAR": ("Bar", "Medium"),
    "PUB": ("Bar", "Medium"),
    "COMIC_BOOK_SHOP": ("Bookstore", "Low"),
    "LUNCH_VAN": ("Fast Food Shop", "Low"),
    "DEPARTMENT_STORE": ("Retail Store", "Medium"),
    "KEBAB_SHOP": ("Fast Food Shop", "Low"),
    "WINE_BAR": ("Bar", "Medium"),
    "ELECTRONICS_SHOP": ("Electronics", "Medium"),
    "RESTAURANT": ("Restaurant", "Medium"),
    "LOCAL_PUB": ("Bar", "Medium"),
    "LUNCH_PLACE": ("Fast Food Shop", "Low"),
    "FASHION_SHOP": ("Retail Store", "Medium"),
    "FASHIONABLE_SPORTSWARE_SHOP": ("Luxury Store", "High"),
    "SCHOOL_SUPPLY_STORE": ("Bookstore", "Low"),
    "LOCAL_BOOKSHOP": ("Bookstore", "Low"),
    "TRAINER_SHOP": ("Retail Store", "Medium"),
    "BOOKSHOP": ("Bookstore", "Low"),
    "KIDS_ACTIVITY_CENTRE": ("Other", "Medium"),
    "VIDEO_GAME_STORE": ("Electronics", "Medium"),
    "CLOTHES_SHOP": ("Retail Store", "Medium"),
    "TAKEAWAY_CURRY": ("Fast Food Shop", "Low"),
    "TECH_SHOP": ("Electronics", "High"),
    "NERDY_BOOK_STORE": ("Bookstore", "Low"),
    "WHISKEY_BAR": ("Bar", "High"),
    "PET_TOY_SHOP": ("Pet Shop", "Low"),
    "DVD_SHOP": ("Electronics", "Low"),
    "CHILDRENDS_SHOP": ("Toy Shop", "Low"),
    "GAME_SHOP": ("Electronics", "Medium"),
    "INDIAN_RESTAURANT": ("Restaurant", "Medium"),
    "COCKTAIL_BAR": ("Bar", "High"),
    "RUNNING_SHOP": ("Retail Store", "Medium"),
    "DIY_STORE": ("Other", "Medium"),
    "COOKSHOP": ("Other", "Medium"),
    "HOME_IMPROVEMENT_STORE": ("Other", "Medium"),
    "PET_SHOP": ("Pet Shop", "Low"),
    "CHINESE_TAKEAWAY": ("Fast Food Shop", "Low"),
    "BUTCHERS": ("Butcher's", "Medium"),
    "SECOND_HAND_BOOKSHOP": ("Bookstore", "Low"),
    "G&T_BAR": ("Bar", "Medium"),
    "GREENGROCER": ("Supermarket", "Low"),
    "JEWLLERY_SHOP": ("Luxury Store", "High"),
    "ACCESSORY_SHOP": ("Retail Store", "Medium"),
    "TAKEAWAY": ("Fast Food Shop", "Low"),
    "KIDS_CLOTHING_SHOP": ("Retail Store", "Low"),
    "SPORT_SHOP": ("Retail Store", "Medium"),
    "STEAK_HOUSE": ("Restaurant", "High"),
    "HIPSTER_ELECTRONICS_SHOP": ("Electronics", "High"),
    "CHINESE_RESTAURANT": ("Restaurant", "Medium"),
    "SEAFOOD_RESAURANT": ("Restaurant", "High"),
    "STREAMING_SERVICE": ("Other", "Medium"),
    "GYM": ("Other", "Medium"),
    "WHISKEY_SHOP": ("Luxury Store", "High"),
    "TEA_SHOP": ("Tea Shop", "Medium"),
    "RESTAURANT_VOUCHER": ("Restaurant", "Medium"),
    "ROASTERIE": ("Coffee Shop", "Medium"),
    "LIQUOR_STORE": ("Bar", "Medium"),
    "WINE_CELLAR": ("Bar", "High"),
    "LARGE_SUPERMARKET": ("Supermarket", "Low"),
    "EXPRESS_SUPERMARKET": ("Supermarket", "Low"),
    "BUTCHER": ("Butcher's", "Medium"),
    "A_SUPERMARKET": ("Supermarket", "Low"),
    "THE_SUPERMARKET": ("Supermarket", "Low")
}

shop_name_to_tag_level = {}
for name, (tag, level) in shop_tags_and_levels.items():
    shop_name_to_tag_level[name] = (tag, level)

def assign_tag_and_level(shop_name):
    return shop_name_to_tag_level.get(shop_name, ("Transfer", "Unknown"))

df['Shop_type'], df['Expected consumption level'] = zip(*df['to_randomly_generated_account'].apply(assign_tag_and_level))

df.to_csv('tagged_data.csv', index=False)



