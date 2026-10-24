"""用户台词与回答模板。

生活事件的台词只描述发生了什么，从不说"我的偏好变了"；agent 要自己推断哪些偏好受影响。
"""

from .schema import PRICE_BANDS

# ---------- 生活事件 ----------
# effects(state, rng) -> {属性: 新值}；pre(state) -> 是否适用
# 台词里的 {new_home}/{new_office} 由事件效果填入。

EVENTS = {
    "fitness": {
        "cn": "开始健身",
        "pre": lambda s: s["diet"] == "普通" and s["cuisine"] != "轻食",
        "effects": lambda s, rng: {"diet": "控碳水", "cuisine": "轻食"},
        "mentions": [
            "最近办了张健身卡，每周练四次，教练把我每天吃什么都盯得很紧。",
            "跟着私教练了一个月了，他说想看到效果，主食那块得先管住。",
        ],
    },
    "move": {
        "cn": "搬家",
        "pre": lambda s: True,
        "effects": "_move",
        "mentions": {
            "开车": ["上周末终于搬到{new_home}的新房子了，那边地铁还没通，现在天天开车。"],
            "骑车": ["刚搬到{new_home}，离公司就两三公里，现在每天骑车上下班。"],
            "地铁": ["搬到{new_home}了，小区门口就是地铁站，现在出门都坐地铁。"],
        },
    },
    "baby": {
        "cn": "家里添了宝宝",
        "pre": lambda s: s["kid"] == "无" and s["evening"] == "可以晚归",
        "effects": lambda s, rng: {"kid": "有婴儿", "evening": "要早回"},
        "mentions": [
            "家里的小闺女刚满月，晚上的哄睡归我管，天天困得不行。",
            "我们家宝宝出生快两个月了，现在全家作息都围着他转。",
        ],
    },
    "dog": {
        "cn": "养了狗",
        "pre": lambda s: s["pet"] == "无" and s["evening"] == "可以晚归",
        "effects": lambda s, rng: {"pet": "有狗", "evening": "要早回"},
        "mentions": [
            "上周从救助站领回来一只柯基，黏人得很，每天晚上都等我回家遛它。",
            "家里新来了一只小金毛，我走到哪都想带着它，晚上还得按点回去遛。",
        ],
    },
    "promotion": {
        "cn": "升职调岗",
        "pre": lambda s: s["budget"] != "高",
        "effects": "_promotion",
        "mentions": [
            "升职了，下周开始调到{new_office}的总部上班，以后请人吃饭也得讲究点了。",
        ],
    },
    "pregnancy": {
        "cn": "怀孕",
        "pre": lambda s: s["spicy"] == "爱吃辣" and s["cuisine"] != "粤菜",
        "effects": lambda s, rng: {"spicy": "不吃辣", "cuisine": "粤菜"},
        "mentions": [
            "最近查出来怀孕了，孕吐挺厉害，医生让我吃得清淡些，生冷刺激的都别碰。",
        ],
    },
    "newjob": {
        "cn": "换工作降薪",
        "pre": lambda s: s["budget"] != "低",
        "effects": "_newjob",
        "mentions": [
            "换到{new_office}一家创业公司了，收入少了一截，最近得精打细算过日子。",
        ],
    },
    "checkup": {
        "cn": "体检血糖偏高",
        "pre": lambda s: s["diet"] == "普通",
        "effects": lambda s, rng: {"diet": "低糖"},
        "mentions": ["上周单位体检，医生说我血糖有点偏高。"],
    },
    "car": {
        "cn": "买车",
        "pre": lambda s: s["commute"] != "开车",
        "effects": lambda s, rng: {"commute": "开车"},
        "mentions": ["刚提了辆新车，周末出门终于不用挤地铁了。"],
    },
}

# ---------- 临时情况（只影响本次任务，不应改写长期记忆）----------
TEMPS = {
    "throat": {
        "cn": "嗓子发炎",
        "attr": "spicy",
        "domains": ["dinner", "lunch"],
        "pre": lambda s: s["spicy"] == "爱吃辣",
        "value": lambda s, rng: "不吃辣",
        "mentions": ["这两天嗓子发炎，这顿先吃点不辣的吧。", "我这周嗓子疼得厉害，这次别选辣的。"],
    },
    "expense": {
        "cn": "公司报销",
        "attr": "budget",
        "domains": ["dinner", "hotel"],
        "pre": lambda s: s["budget"] != "高",
        "value": lambda s, rng: "高",
        "mentions": ["这次是招待客户，公司报销，档次高一点没关系。"],
    },
    "frugal": {
        "cn": "这顿想省点",
        "attr": "budget",
        "domains": ["dinner", "lunch"],
        "pre": lambda s: s["budget"] != "低",
        "value": lambda s, rng: "低",
        "mentions": ["月底了手头有点紧，这顿就吃便宜点的。"],
    },
    "elsewhere": {
        "cn": "这次换个区聚",
        "attr": "home",
        "domains": ["dinner"],
        "pre": lambda s: True,
        "value": "_other_area",
        "mentions": ["这次聚会定在{temp_value}，朋友们都住那边，就在那附近找吧。"],
    },
}

# ---------- 任务请求 ----------
REQUESTS = {
    "dinner": [
        "这周六晚上想约两个朋友在家附近吃顿饭，帮我挑一家。",
        "周五请同事吃饭，就在我家附近，帮我订个地方。",
        "周末想跟老同学在我家那边聚一下，帮我选家店。",
    ],
    "lunch": [
        "帮我在公司附近找个工作日能常去吃午饭的地方。",
        "中午想在公司附近吃，帮我挑一家。",
    ],
    "hotel": [
        "下个月想去{city}玩两天，帮我订个酒店。",
        "月底要去{city}过个周末，帮我挑家酒店。",
    ],
}
CITIES = ["苏州", "南京", "上海", "黄山", "宁波"]

# ---------- 选项名称 ----------
DINING_NAMES = {
    "面食": ["老街面馆", "一碗好面", "葱油面铺", "西北面庄", "阿婆手擀面"],
    "火锅": ["蜀香火锅", "牛油锅子", "老灶火锅", "椰子鸡火锅", "围炉小馆"],
    "日料": ["鮨一", "松叶寿司", "居酒屋小町", "海之味料理", "樱庭"],
    "粤菜": ["点都德茶楼", "粤香园", "陶陶居", "广府小厨", "早茶记"],
    "川菜": ["川味坊", "巴蜀人家", "辣子鸡馆", "麻婆小院", "蓉城记"],
    "轻食": ["绿洲沙拉", "能量碗", "轻体厨房", "麦田轻食", "元气碗"],
    "西餐": ["橡树牛排", "托斯卡纳", "小酒馆", "布鲁克林餐厅", "白鸽西餐"],
}
HOTEL_NAMES = ["云栖酒店", "湖畔居", "城市之光酒店", "松间民宿", "望江酒店", "梧桐里酒店",
               "悦榕小筑", "山舍", "青藤客栈", "雅致酒店", "星河酒店", "竹影民宿"]


# ---------- 回答模板 ----------
def _budget_words(band):
    d, l, h = (PRICE_BANDS[k][band] for k in ("dinner", "lunch", "hotel"))
    head = {"低": "我花钱比较省", "中": "预算中等就好", "高": "我不太在意价格，太便宜的反而不想去"}[band]
    return f"{head}：晚饭人均 {d[0]}–{d[1]} 元，午饭 {l[0]}–{l[1]} 元，酒店一晚 {h[0]}–{h[1]} 元"


def answer_attr(attr, long_val, eff_val, temp_cn=None):
    """回答单个属性。eff_val 与 long_val 不同说明有临时情况。"""
    def say(v):
        return {
            "allergy": lambda v: "我没有过敏的东西" if v == "无" else f"我对{v}过敏，这个一定要避开",
            "diet": lambda v: {"普通": "饮食上没什么特别限制", "素食": "我吃素",
                               "控碳水": "我现在在控制碳水，得选有低碳水菜品的",
                               "低糖": "我得少吃糖，要选有低糖菜品的"}[v],
            "cuisine": lambda v: f"我最爱吃{v}",
            "spicy": lambda v: "我无辣不欢，得有辣的" if v == "爱吃辣" else "我不吃辣，得是清淡口味的",
            "budget": _budget_words,
            "home": lambda v: f"我住在{v}",
            "office": lambda v: f"我在{v}上班",
            "commute": lambda v: {"地铁": "我平时坐地铁", "骑车": "我一般骑车",
                                  "开车": "我开车出门，得有停车位"}[v],
            "kid": lambda v: "家里没有小孩" if v == "无" else "家里有个小宝宝，吃饭得有宝宝椅，住酒店要婴儿床",
            "pet": lambda v: "没养宠物" if v == "无" else "我养了只狗，出去玩都带着它，酒店得能带宠物",
            "evening": lambda v: "晚上几点回家都行" if v == "可以晚归" else "晚上得早点回家，晚饭只能订 18:30 以前的",
        }[attr](v)

    if eff_val == long_val:
        return say(long_val)
    if attr == "home":
        return f"{say(long_val)}，不过这次聚会定在{eff_val}，就在那边找"
    return f"平时{say(long_val)}；不过这次{temp_cn}，{say(eff_val)}"


NO_TOPIC_ANSWER = "嗯？这个我没什么特别的想法，你看着办就行。"
TOO_MANY_ANSWER = "一下问太多了，其他的回头再说吧。"
NOTHING_NEW = "最近挺好的，没什么特别的变化。"
