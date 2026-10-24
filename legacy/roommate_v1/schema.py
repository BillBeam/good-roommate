"""偏好属性、提问话题与任务领域。

用户的真实状态是一个 {属性: 取值} 字典，只由程序维护，agent 看不到。
"""

AREAS = ["西湖", "滨江", "拱墅", "余杭", "上城"]
CUISINES = ["面食", "火锅", "日料", "粤菜", "川菜", "轻食", "西餐"]
ALLERGENS = ["花生", "海鲜", "乳制品"]
DIET_TAG = {"素食": "素食友好", "控碳水": "低碳水", "低糖": "低糖"}

# 属性 -> 可取值
ATTRS = {
    "allergy": ["无", "花生", "海鲜", "乳制品"],
    "diet": ["普通", "素食", "控碳水", "低糖"],
    "cuisine": CUISINES,
    "spicy": ["爱吃辣", "不吃辣"],
    "budget": ["低", "中", "高"],
    "home": AREAS,
    "office": AREAS,
    "commute": ["地铁", "骑车", "开车"],
    "kid": ["无", "有婴儿"],
    "pet": ["无", "有狗"],
    "evening": ["可以晚归", "要早回"],
}

ATTR_CN = {
    "allergy": "过敏", "diet": "饮食方式", "cuisine": "爱吃的菜系", "spicy": "吃辣",
    "budget": "预算档次", "home": "住处", "office": "公司位置", "commute": "出行方式",
    "kid": "小孩", "pet": "宠物", "evening": "晚上作息",
}

# 提问话题 -> 它覆盖的属性。用户一次最多回答两个话题。
TOPICS = {
    "饮食": ["allergy", "diet"],
    "口味": ["cuisine", "spicy"],
    "预算": ["budget"],
    "住处": ["home"],
    "公司": ["office"],
    "出行": ["commute"],
    "家庭": ["kid", "pet", "evening"],
    "近况": [],
}
MAX_TOPICS_PER_QUESTION = 2

# 任务领域 -> 判分时用到的属性
DOMAINS = {
    "dinner": ["allergy", "diet", "cuisine", "spicy", "budget", "home", "commute", "kid", "evening"],
    "lunch": ["allergy", "diet", "cuisine", "spicy", "budget", "office"],
    "hotel": ["budget", "diet", "commute", "kid", "pet"],
}
DOMAIN_CN = {"dinner": "晚餐", "lunch": "午餐", "hotel": "酒店", "util": "与偏好无关"}

# 预算档次在各领域对应的价格区间（档次之间留空隙，避免歧义）
PRICE_BANDS = {
    "dinner": {"低": (40, 80), "中": (90, 150), "高": (180, 300)},
    "lunch": {"低": (15, 28), "中": (32, 50), "高": (58, 90)},
    "hotel": {"低": (250, 400), "中": (450, 800), "高": (900, 1500)},
}

EARLY_SLOTS = ["17:30", "18:00"]
LATE_SLOTS = ["20:00", "20:30"]


def topics_for_domain(domain):
    attrs = set(DOMAINS.get(domain, []))
    return [t for t, a in TOPICS.items() if attrs & set(a)]


def topic_of_attr(attr):
    for t, a in TOPICS.items():
        if attr in a:
            return t
    raise KeyError(attr)
