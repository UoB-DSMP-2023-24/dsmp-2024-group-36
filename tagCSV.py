import pandas as pd


csv_file_path = 'fake_transactional_data_24.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 店铺标签和等级的字典
shop_tags_and_levels = {
    "CINEMA": "Cinema",
    "HIPSTER_COFFEE_SHOP": "Coffee Shop",
    "TOTALLY_A_REAL_COFFEE_SHOP": "Coffee Shop",
    "COFFEE_SHOP": "Coffee Shop",
    "CAFE": "Cafe",
    "A_CAFE": "Cafe",
    "LOCAL_RESTAURANT": "Restaurant",
    "A_LOCAL_COFFEE_SHOP": "Coffee Shop",
    "GOURMET_COFFEE_SHOP": "High-End Coffee Shop",
    "LOCAL_WATERING_HOLE": "Bar",
    "SANDWICH_SHOP": "Fast Food Shop",
    "TOY_SHOP": "Toy Shop",
    "PRETENTIOUS_COFFEE_SHOP": "High-End Coffee Shop",
    "BAR": "Bar",
    "PUB": "Bar",
    "COMIC_BOOK_SHOP": "Bookstore",
    "LUNCH_VAN": "Fast Food Shop",
    "DEPARTMENT_STORE": "Retail Store",
    "KEBAB_SHOP": "Fast Food Shop",
    "WINE_BAR": "Bar",
    "ELECTRONICS_SHOP": "Electronics",
    "RESTAURANT": "Restaurant",
    "LOCAL_PUB": "Bar",
    "LUNCH_PLACE": "Fast Food Shop",
    "FASHION_SHOP": "Retail Store",
    "FASHIONABLE_SPORTSWARE_SHOP": "Luxury Store",
    "SCHOOL_SUPPLY_STORE": "Bookstore",
    "LOCAL_BOOKSHOP": "Bookstore",
    "TRAINER_SHOP": "Retail Store",
    "BOOKSHOP": "Bookstore",
    "KIDS_ACTIVITY_CENTRE": "Other",
    "VIDEO_GAME_STORE": "Electronics",
    "CLOTHES_SHOP": "Retail Store",
    "TAKEAWAY_CURRY": "Fast Food Shop",
    "TECH_SHOP": "Electronics",
    "NERDY_BOOK_STORE": "Bookstore",
    "WHISKEY_BAR": "Bar",
    "PET_TOY_SHOP": "Pet Shop",
    "DVD_SHOP": "Electronics",
    "CHILDRENDS_SHOP": "Toy Shop",
    "GAME_SHOP": "Electronics",
    "INDIAN_RESTAURANT": "Restaurant",
    "COCKTAIL_BAR": "Bar",
    "RUNNING_SHOP": "Retail Store",
    "DIY_STORE": "Other",
    "COOKSHOP": "Other",
    "HOME_IMPROVEMENT_STORE": "Other",
    "PET_SHOP": "Pet Shop",
    "CHINESE_TAKEAWAY": "Fast Food Shop",
    "BUTCHERS": "Butcher's",
    "SECOND_HAND_BOOKSHOP": "Bookstore",
    "G&T_BAR": "Bar",
    "GREENGROCER": "Supermarket",
    "JEWLLERY_SHOP": "Luxury Store",
    "ACCESSORY_SHOP": "Retail Store",
    "TAKEAWAY": "Fast Food Shop",
    "KIDS_CLOTHING_SHOP": "Retail Store",
    "SPORT_SHOP": "Retail Store",
    "STEAK_HOUSE": "Restaurant",
    "HIPSTER_ELECTRONICS_SHOP": "Electronics",
    "CHINESE_RESTAURANT": "Restaurant",
    "SEAFOOD_RESAURANT": "Restaurant",
    "STREAMING_SERVICE": "Other",
    "GYM": "Other",
    "WHISKEY_SHOP": "Luxury Store",
    "TEA_SHOP": "Tea Shop",
    "RESTAURANT_VOUCHER": "Restaurant",
    "ROASTERIE": "Coffee Shop",
    "LIQUOR_STORE": "Bar",
    "WINE_CELLAR": "Bar",
    "LARGE_SUPERMARKET": "Supermarket",
    "EXPRESS_SUPERMARKET": "Supermarket",
    "BUTCHER": "Butcher's",
    "A_SUPERMARKET": "Supermarket",
    "THE_SUPERMARKET": "Supermarket"
}

# 更新DataFrame
df['Shop_type'] = df['to_randomly_generated_account'].apply(lambda x: shop_tags_and_levels.get(x, "Transfer"))

# 输出更新后的DataFrame
print(df.head())

# 保存到新的CSV文件
df.to_csv('tagged_data.csv', index=False)





