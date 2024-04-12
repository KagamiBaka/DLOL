import json


files = ['test.json', 'train.json']
labels_full_names = {
    "naphtha": "Naphtha",
    "lead": "Lead",
    "zinc": "Zinc",
    "gas": "Natural Gas",
    "potato": "Potato",
    "alum": "Aluminum",
    "corn": "Corn",
    "heat": "Heating Oil",
    "groundnut": "Groundnut",
    "cotton": "Cotton",
    "earn": "Earnings",
    "coffee": "Coffee",
    "carcass": "Carcass",
    "soy-oil": "Soybean Oil",
    "bop": "Balance of Payments",
    "wheat": "Wheat",
    "reserves": "Reserves",
    "rape-oil": "Rapeseed Oil",
    "soy-meal": "Soybean Meal",
    "lumber": "Lumber",
    "wpi": "Wholesale Price Index",
    "gold": "Gold",
    "nickel": "Nickel",
    "lin-oil": "Linseed Oil",
    "l-cattle": "Live Cattle",
    "sunseed": "Sunflower Seed",
    "rand": "South African Rand",
    "lei": "Lei",
    "rye": "Rye",
    "fuel": "Fuel",
    "income": "Income",
    "iron-steel": "Iron and Steel",
    "palmkernel": "Palm Kernel",
    "hog": "Hog",
    "jet": "Jet Fuel",
    "soybean": "Soybean",
    "barley": "Barley",
    "sorghum": "Sorghum",
    "jobs": "Jobs",
    "sugar": "Sugar",
    "money-supply": "Money Supply",
    "rapeseed": "Rapeseed",
    "palm-oil": "Palm Oil",
    "rice": "Rice",
    "palladium": "Palladium",
    "money-fx": "Money Foreign Exchange",
    "nzdlr": "New Zealand Dollar",
    "livestock": "Livestock",
    "veg-oil": "Vegetable Oil",
    "gnp": "Gross National Product",
    "orange": "Orange",
    "cpi": "Consumer Price Index",
    "cotton-oil": "Cottonseed Oil",
    "yen": "Japanese Yen",
    "propane": "Propane",
    "acq": "Acquisitions",
    "castor-oil": "Castor Oil",
    "coconut": "Coconut",
    "oilseed": "Oilseed",
    "nkr": "Norwegian Krone",
    "meal-feed": "Meal Feed",
    "tea": "Tea",
    "sun-meal": "Sunflower Meal",
    "ipi": "Industrial Production Index",
    "pet-chem": "Petrochemicals",
    "instal-debt": "Installment Debt",
    "dfl": "Dutch Florin",
    "trade": "Trade",
    "grain": "Grain",
    "rubber": "Rubber",
    "sun-oil": "Sunflower Oil",
    "cpu": "Capacity Utilization",
    "dmk": "Deutsche Mark",
    "retail": "Retail",
    "cocoa": "Cocoa",
    "coconut-oil": "Coconut Oil",
    "ship": "Shipping",
    "nat-gas": "Natural Gas",
    "platinum": "Platinum",
    "tin": "Tin",
    "interest": "Interest Rates",
    "copper": "Copper",
    "strategic-metal": "Strategic Metals",
    "silver": "Silver",
    "copra-cake": "Copra Cake",
    "oat": "Oat",
    "housing": "Housing",
    "groundnut-oil": "Groundnut Oil",
    "dlr": "Dollar",
    "crude": "Crude Oil"
} # generated by GPT-4


for file in files:
    unique_labels = set()
    a = open(file)
    dics = []
    for line in a.readlines():
        dic = json.loads(line)
        dics += [dic]

    english_prompt = "Classify the following Reuters economic and trade news as the correct topic: "

    # 处理每一行
    processed_data = []
    for data in dics:
        
        # 在"text"字段前加上英文提示
        data["text"] = english_prompt + data["text"]
        
        # 收集独特的标签
        if data["label"][0] == " ":
            labels = data["label"][1:].split(", ")
        else:
            labels = data["label"].split(", ")
        new_label = ','.join([" "+ labels_full_names[i].lower() for i in labels])
        data['label'] = new_label
        # 将处理后的数据转换回JSON字符串（如果需要保存回文件）
        processed_data.append(data)
    import random
    random.seed(42)
    random.shuffle(processed_data)
    if file == 'test.json':
        with open('shuffled/' + 'test.json', 'w') as f:
            for data in processed_data[0:2005]:
                json.dump(data, f)
                f.write('\n')
        with open('shuffled/' + 'dev.json', 'w') as f:
            for data in processed_data[2005:]:
                json.dump(data, f)
                f.write('\n')
    else:
        with open('shuffled/' + file, 'w') as f:
            for data in processed_data:
                json.dump(data, f)
                f.write('\n')


print("Unique labels:", unique_labels)
print(len(unique_labels))