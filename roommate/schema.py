"""偏好属性与任务领域（第二版）。

用户的真实状态是一个 {属性: 取值} 字典，只由程序维护，agent 看不到。
agent 提问时要声明问的是哪几项属性（ATTR_CN 里的中文名），一次最多 MAX_ATTRS_PER_QUESTION 项；
模拟用户只回答声明的这几项。
"""

AREAS = ["西湖", "滨江", "拱墅", "余杭", "上城"]
CUISINES = ["面食", "火锅", "日料", "粤菜", "川菜", "轻食", "西餐"]
ALLERGENS = ["花生", "海鲜", "乳制品"]
DIET_TAG = {"素食": "素食友好", "控碳水": "低碳水", "低糖": "低糖"}
DIET_TAGS = list(DIET_TAG.values())

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
    "evening": ["下班晚", "要早回"],
}
ATTR_ORDER = list(ATTRS)

# agent 提问时用的属性名
ATTR_CN = {
    "allergy": "过敏", "diet": "饮食限制", "cuisine": "菜系", "spicy": "辣度",
    "budget": "预算", "home": "住处", "office": "公司", "commute": "出行",
    "kid": "小孩", "pet": "宠物", "evening": "晚归",
}
CN_ATTR = {v: k for k, v in ATTR_CN.items()}
MAX_ATTRS_PER_QUESTION = 2

# 两向属性：每个取值都要求选项有对应的特征（选项的特征和用户的取值必须一致）。
# 其余是单向属性：只有部分取值带约束（如"有婴儿"要宝宝椅），"无"之类的取值对选项没有要求。
TWO_SIDED = {"cuisine", "spicy", "budget", "home", "office", "evening"}

# 任务领域 -> 判分时用到的属性。第二版去掉了酒店（原因见报告"已知局限"）。
DOMAINS = {
    "dinner": ["allergy", "diet", "cuisine", "spicy", "budget", "home", "commute", "kid", "pet", "evening"],
    "lunch": ["allergy", "diet", "cuisine", "spicy", "budget", "office"],
}
DOMAIN_CN = {"dinner": "晚餐", "lunch": "午餐", "util": "与偏好无关"}

# 预算档次在各领域对应的价格区间（档次之间留空隙，避免歧义）
PRICE_BANDS = {
    "dinner": {"低": (40, 80), "中": (90, 150), "高": (180, 300)},
    "lunch": {"低": (15, 28), "中": (32, 50), "高": (58, 90)},
}

EARLY_SLOTS = ["17:30", "18:00"]
LATE_SLOTS = ["20:00", "20:30"]
EARLY_CUTOFF = "18:30"   # 要早回：只能订这之前的
LATE_FROM = "19:30"      # 下班晚：只能订这之后的


def area_attr(domain):
    return "home" if domain == "dinner" else "office"
