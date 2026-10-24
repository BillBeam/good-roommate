"""用户台词与回答模板（第二版）。

- 生活事件的台词只描述发生了什么，从不说"我的偏好变了"；但事件改动的每一项属性都能从台词里读出来：
  CUES 给每条台词标注了每项改动是"明说"还是"可推断"（推不出来的改动已经删掉）。
- 回答由"短句"拼成，每个短句对应一条可核对的陈述（PHRASES）。parse_claims 只看文字、
  按同一份短句表反解出陈述，用来检验回答是否和用户的真实状态、和用户自己说过的话一致。
"""

import re

from .schema import ALLERGENS, AREAS, ATTR_CN, CUISINES, PRICE_BANDS

# ---------- 生活事件 ----------
# 每个事件：适用条件 pre(state)、改动 effects(state, rng) -> {属性: 新值}、台词 mentions。
# 每条台词带 cues：{属性: "明说" | "可推断"}，覆盖该事件可能改动的全部属性。
# 每个事件至少改动一项两向属性（见 schema.TWO_SIDED），保证新旧状态在选项上互相排斥。

EVENTS = {
    "fitness": {
        "cn": "开始健身",
        "pre": lambda s: s["diet"] == "普通" and s["cuisine"] != "轻食",
        "mentions": [
            ("最近办了张健身卡，教练让我把主食管住，现在基本天天吃轻食沙拉。",
             {"diet": "可推断", "cuisine": "明说"}),
            ("跟着私教练了一个月，他让我少吃米饭面条这些碳水，我现在最常吃的就是轻食了。",
             {"diet": "明说", "cuisine": "明说"}),
        ],
    },
    "move": {
        "cn": "搬家",
        "pre": lambda s: True,
        "mentions": {
            "开车": [("上周末搬到{new_home}的新房子了，那边地铁还没通，现在天天开车出门。",
                      {"home": "明说", "commute": "明说"})],
            "骑车": [("刚搬到{new_home}，跟公司在一个区，现在每天骑车上下班。",
                      {"home": "明说", "commute": "明说"})],
            "地铁": [("搬到{new_home}了，小区门口就是地铁站，现在出门都坐地铁。",
                      {"home": "明说", "commute": "明说"})],
        },
    },
    "baby": {
        "cn": "家里添了宝宝",
        "pre": lambda s: s["kid"] == "无" and s["evening"] == "下班晚",
        "mentions": [
            ("家里的小闺女刚满月，晚上的哄睡归我管，下了班就得赶紧回家。",
             {"kid": "明说", "evening": "可推断"}),
            ("我们家宝宝出生快两个月了，现在全家作息都围着他转，晚上不能在外面待太晚。",
             {"kid": "明说", "evening": "可推断"}),
        ],
    },
    "dog": {
        "cn": "养了狗",
        "pre": lambda s: s["pet"] == "无" and s["evening"] == "下班晚",
        "mentions": [
            ("上周从救助站领回来一只柯基，黏人得很，去餐厅吃饭我都想带上它；晚上也得早点回家遛它，不能在外面待太晚。",
             {"pet": "明说", "evening": "明说"}),
        ],
    },
    "promotion": {
        "cn": "升职调岗",
        "pre": lambda s: s["budget"] != "高",
        "mentions": [
            ("升职了，下周开始调到{new_office}的总部上班，收入涨了一大截，以后吃饭就挑高档的地方，不用在意价格了。",
             {"budget": "明说", "office": "明说"}),
        ],
    },
    "pregnancy": {
        "cn": "怀孕",
        "pre": lambda s: s["spicy"] == "爱吃辣" and s["cuisine"] != "粤菜",
        "mentions": [
            ("最近查出来怀孕了，医生让我吃得清淡些，辣的一口都不能碰，现在只想吃粤菜那种清淡的。",
             {"spicy": "明说", "cuisine": "明说"}),
        ],
    },
    "newjob": {
        "cn": "换工作降薪",
        "pre": lambda s: s["budget"] != "低",
        "mentions": [
            ("换到{new_office}一家创业公司了，收入少了一截，最近吃饭都得挑便宜的。",
             {"budget": "可推断", "office": "明说"}),
        ],
    },
    "checkup": {
        "cn": "体检血糖偏高",
        "pre": lambda s: s["diet"] == "普通" and s["spicy"] == "爱吃辣",
        "mentions": [
            ("上周单位体检，医生说我血糖偏高、胃也不太好，让我少吃甜的，辣的也别吃了。",
             {"diet": "可推断", "spicy": "明说"}),
        ],
    },
}

# 因健康原因不吃辣的事件：被问到"饮食限制"时必须说出来
HEALTH_SPICY_EVENTS = {"pregnancy", "checkup"}


def other(rng, values, cur):
    return rng.choice([v for v in values if v != cur])


def event_effects(etype, s, rng):
    """事件对状态的改动 {属性: 新值}（只含真正变了的属性）。"""
    if etype == "fitness":
        ch = {"diet": "控碳水", "cuisine": "轻食"}
    elif etype == "move":
        new_home = other(rng, AREAS, s["home"])
        mode = "骑车" if new_home == s["office"] else "开车" if new_home == "余杭" else "地铁"
        ch = {"home": new_home, "commute": mode}
    elif etype in ("baby",):
        ch = {"kid": "有婴儿", "evening": "要早回"}
    elif etype == "dog":
        ch = {"pet": "有狗", "evening": "要早回"}
    elif etype == "promotion":
        ch = {"budget": "高", "office": other(rng, AREAS, s["office"])}
    elif etype == "newjob":
        ch = {"budget": "低", "office": other(rng, AREAS, s["office"])}
    elif etype == "pregnancy":
        ch = {"spicy": "不吃辣", "cuisine": "粤菜"}
    elif etype == "checkup":
        ch = {"diet": "低糖", "spicy": "不吃辣"}
    else:
        raise KeyError(etype)
    return {a: v for a, v in ch.items() if s[a] != v}


def event_mention(etype, after, rng):
    """返回 (台词, cues)。after 是事件之后的状态。"""
    ev = EVENTS[etype]
    if etype == "move":
        mode = "骑车" if after["home"] == after["office"] else "开车" if after["home"] == "余杭" else "地铁"
        tpl, cues = rng.choice(ev["mentions"][mode])
    else:
        tpl, cues = rng.choice(ev["mentions"])
    return tpl.format(new_home=after["home"], new_office=after["office"]), dict(cues)


# ---------- 临时情况（只影响本次任务，不应改写长期记忆）----------
TEMPS = {
    "throat": {
        "cn": "嗓子发炎", "reason": "嗓子发炎", "attr": "spicy", "domains": ["dinner", "lunch"],
        "pre": lambda s: s["spicy"] == "爱吃辣",
        "mentions": ["这两天嗓子发炎，这顿先吃点不辣的吧。", "我这周嗓子疼得厉害，这次别选辣的。"],
    },
    "expense": {
        "cn": "公司报销", "reason": "公司报销", "attr": "budget", "domains": ["dinner", "lunch"],
        "pre": lambda s: s["budget"] != "高",
        "mentions": ["这次是招待客户，公司报销，得订个高档点的地方。"],
    },
    "frugal": {
        "cn": "这顿想省点", "reason": "手头有点紧", "attr": "budget", "domains": ["dinner", "lunch"],
        "pre": lambda s: s["budget"] != "低",
        "mentions": ["月底了手头有点紧，这顿就吃便宜点的。"],
    },
    "elsewhere": {
        "cn": "这次换个区聚", "reason": "换个区聚", "attr": "home", "domains": ["dinner"],
        "pre": lambda s: True,
        "mentions": ["这次聚会定在{temp_value}，朋友们都住那边，就在那附近找吧。"],
    },
}


def temp_value(ttype, s, rng):
    return {"throat": "不吃辣", "expense": "高", "frugal": "低"}.get(ttype) or other(rng, AREAS, s["home"])


# ---------- 任务请求 ----------
REQUESTS = {
    "dinner": [
        "这周六晚上想约两个朋友在家附近吃顿饭，帮我挑一家。",
        "周五请同事吃饭，就在我家附近，帮我订个地方。",
        "周末想跟老同学在我家那边聚一下，帮我选家店。",
    ],
    "dinner_elsewhere": ["周末想跟老同学聚一下，帮我选家店。", "这周六约了朋友吃晚饭，帮我订个地方。"],
    "lunch": [
        "帮我在公司附近找个工作日能常去吃午饭的地方。",
        "中午想在公司附近吃，帮我挑一家。",
    ],
}

# ---------- 选项名称 ----------
DINING_NAMES = {
    "面食": ["老街面馆", "一碗好面", "葱油面铺", "西北面庄", "阿婆手擀面", "面面俱到"],
    "火锅": ["蜀香火锅", "牛油锅子", "老灶火锅", "椰子鸡火锅", "围炉小馆", "一锅鲜"],
    "日料": ["鮨一", "松叶寿司", "居酒屋小町", "海之味料理", "樱庭", "竹笙割烹"],
    "粤菜": ["点都德茶楼", "粤香园", "陶陶居", "广府小厨", "早茶记", "岭南人家"],
    "川菜": ["川味坊", "巴蜀人家", "辣子鸡馆", "麻婆小院", "蓉城记", "锦江小馆"],
    "轻食": ["绿洲沙拉", "能量碗", "轻体厨房", "麦田轻食", "元气碗", "青禾"],
    "西餐": ["橡树牛排", "托斯卡纳", "小酒馆", "布鲁克林餐厅", "白鸽西餐", "橄榄树"],
}


# ---------- 回答短句 ----------
def budget_words(band):
    d, l = (PRICE_BANDS[k][band] for k in ("dinner", "lunch"))
    head = {"低": "我花钱比较省", "中": "预算中等就好", "高": "我不太在意价格，太便宜的反而不想去"}[band]
    return f"{head}：晚饭人均 {d[0]}–{d[1]} 元，午饭 {l[0]}–{l[1]} 元"


SPICY_REASON = {
    "pregnancy": "怀孕以后医生让吃清淡",
    "checkup": "医生说我胃不好",
}

# 属性 -> 取值 -> 短句（长期状态）。辣度的"不吃辣"按原因分成几种说法。
PHRASES = {
    "allergy": {"无": "我没有过敏的东西", **{a: f"我对{a}过敏，这个一定要避开" for a in ALLERGENS}},
    "diet": {"普通": "我不吃素，也不用控碳水或控糖", "素食": "我吃素，得选有素食菜品的",
             "控碳水": "我现在在控制碳水，得选有低碳水菜品的", "低糖": "我得少吃糖，要选有低糖菜品的"},
    "cuisine": {c: f"我最爱吃{c}" for c in CUISINES},
    "spicy": {"爱吃辣": "我无辣不欢，得有辣的", "不吃辣": "我不吃辣，得是清淡口味的"},
    "budget": {b: budget_words(b) for b in ("低", "中", "高")},
    "home": {a: f"我住在{a}" for a in AREAS},
    "office": {a: f"我在{a}上班" for a in AREAS},
    "commute": {"地铁": "我平时坐地铁", "骑车": "我一般骑车", "开车": "我开车出门，得有停车位"},
    "kid": {"无": "家里没有小孩", "有婴儿": "家里有个小宝宝，吃饭得有宝宝椅"},
    "pet": {"无": "没养宠物", "有狗": "我养了只狗，出门吃饭都带着它，得选能带宠物的店"},
    "evening": {"下班晚": "我下班晚，晚饭只能订 19:30 以后的", "要早回": "晚上得早点回家，晚饭只能订 18:30 以前的"},
}
SPICY_HEALTH_PHRASE = "我现在不能吃辣（{reason}），得是清淡口味的"
DIET_HEALTH_APPENDIX = "另外{reason}，辣的不能吃"           # 长期健康原因（怀孕、体检）
DIET_TEMP_APPENDIX = "另外这次{temp_cn}，不能吃辣"            # 本次临时健康原因（嗓子发炎）
TOLD_BEFORE = "关于{attr_cn}，这个我之前跟你说过了"
TEMP_LONG_TOLD = "{attr_cn}平时的情况我之前跟你说过了"
TEMP_PART = "不过这次{temp_cn}，{phrase}"
HOME_TEMP_PART = "不过这次聚会定在{area}，就在那边找"

# 全称否定式的短句：只在这一方面确实没有任何约束时才能出现（由测试逐条检查）
BLANKET = {
    "我没有过敏的东西": lambda s: s["allergy"] == "无",
    "家里没有小孩": lambda s: s["kid"] == "无",
    "没养宠物": lambda s: s["pet"] == "无",
}

TOO_MANY_ANSWER = "一下问太多了，其他的回头再说吧。"
NO_ATTR_ERROR = ("错误：没有声明要问哪几项。请在 attributes 里写 1–2 项，可选："
                 + "、".join(ATTR_CN.values()) + "。这次提问没有发出，不扣分。")


def phrase(attr, value, spicy_reason=None):
    if attr == "spicy" and value == "不吃辣" and spicy_reason in SPICY_REASON:
        return SPICY_HEALTH_PHRASE.format(reason=SPICY_REASON[spicy_reason])
    return PHRASES[attr][value]


# ---------- 反解：只看文字，按短句表找出陈述 ----------
def _inventory():
    inv = []
    for attr, table in PHRASES.items():
        for v, p in table.items():
            inv.append((p, attr, v))
    for reason, r in SPICY_REASON.items():
        inv.append((SPICY_HEALTH_PHRASE.format(reason=r), "spicy", "不吃辣"))
        inv.append((DIET_HEALTH_APPENDIX.format(reason=r), "spicy", "不吃辣"))
    inv.sort(key=lambda x: -len(x[0]))  # 长的先匹配，避免"我不吃辣"吃掉更长的句子
    return inv


_INV = _inventory()
_TOLD_RE = re.compile(TOLD_BEFORE.format(attr_cn="(.+?)"))
_TEMP_LONG_TOLD_RE = re.compile(TEMP_LONG_TOLD.format(attr_cn="(.+?)"))
_TEMP_APPX_RE = re.compile(DIET_TEMP_APPENDIX.format(temp_cn="(.+?)"))
_HOME_TEMP_RE = re.compile(HOME_TEMP_PART.format(area="(" + "|".join(AREAS) + ")"))
_TEMP_RE = re.compile(r"^不过这次(.+?)，(.+)$")
_CN_ATTR = {v: k for k, v in ATTR_CN.items()}


def _match_phrase(unit):
    for p, attr, v in _INV:
        if unit == p:
            return attr, v
    return None


def parse_claims(text):
    """把一段回答反解成陈述列表。每条陈述：
    {"kind": "value", "attr", "value", "scope": "long"|"temp"} 或 {"kind": "told", "attr"}。
    遇到短句表里没有的片段就记一条 {"kind": "unknown", "text": ...}（测试要求不出现）。"""
    out = []
    text = text.replace(TOO_MANY_ANSWER, "")
    for unit in [u for u in re.split(r"[；。]", text) if u.strip()]:
        unit = unit.strip()
        if unit.startswith("平时"):
            unit = unit[2:]
            scope = "long"
        else:
            scope = None
        m = _TOLD_RE.fullmatch(unit)
        if m and m.group(1) in _CN_ATTR:
            out.append({"kind": "told", "attr": _CN_ATTR[m.group(1)]})
            continue
        m = _TEMP_LONG_TOLD_RE.fullmatch(unit)
        if m and m.group(1) in _CN_ATTR:
            out.append({"kind": "told", "attr": _CN_ATTR[m.group(1)]})
            continue
        m = _TEMP_APPX_RE.fullmatch(unit)
        if m:
            out.append({"kind": "value", "attr": "spicy", "value": "不吃辣", "scope": "temp"})
            continue
        # "我住在X，不过这次聚会定在Y，就在那边找" 或 "住处平时……，不过这次……"
        m = _HOME_TEMP_RE.search(unit)
        if m:
            head = unit[:m.start()].rstrip("，")
            if head:
                hm = _match_phrase(head)
                hm2 = _TEMP_LONG_TOLD_RE.fullmatch(head)
                if hm:
                    out.append({"kind": "value", "attr": hm[0], "value": hm[1], "scope": "long"})
                elif hm2 and hm2.group(1) in _CN_ATTR:
                    out.append({"kind": "told", "attr": _CN_ATTR[hm2.group(1)]})
                else:
                    out.append({"kind": "unknown", "text": head})
            out.append({"kind": "value", "attr": "home", "value": m.group(1), "scope": "temp"})
            continue
        m = _TEMP_RE.fullmatch(unit)
        if m:
            hm = _match_phrase(m.group(2))
            if hm:
                out.append({"kind": "value", "attr": hm[0], "value": hm[1], "scope": "temp"})
            else:
                out.append({"kind": "unknown", "text": unit})
            continue
        hm = _match_phrase(unit)
        if hm:
            out.append({"kind": "value", "attr": hm[0], "value": hm[1], "scope": scope or "long"})
            continue
        out.append({"kind": "unknown", "text": unit})
    return out


def template_inventory():
    """报告附录用：全部回答模板。"""
    rows = []
    for attr, table in PHRASES.items():
        for v, p in table.items():
            rows.append((ATTR_CN[attr], v, p))
    for reason, r in SPICY_REASON.items():
        rows.append(("辣度", f"不吃辣（健康原因：{EVENTS[reason]['cn']}）", SPICY_HEALTH_PHRASE.format(reason=r)))
        rows.append(("饮食限制（附加）", f"健康原因不吃辣：{EVENTS[reason]['cn']}", DIET_HEALTH_APPENDIX.format(reason=r)))
    rows.append(("饮食限制（附加）", "本次临时不吃辣", DIET_TEMP_APPENDIX.format(temp_cn="{临时情况}")))
    rows.append(("任意", "当前值等于上次告诉 agent 的值", TOLD_BEFORE.format(attr_cn="{属性}")))
    rows.append(("任意（临时情况）", "长期值已告诉过", TEMP_LONG_TOLD.format(attr_cn="{属性}") + "；" +
                 TEMP_PART.format(temp_cn="{临时情况}", phrase="{临时值的短句}")))
    rows.append(("任意（临时情况）", "长期值没告诉过", "平时{长期值的短句}；" +
                 TEMP_PART.format(temp_cn="{临时情况}", phrase="{临时值的短句}")))
    rows.append(("住处（临时情况）", "这次换个区聚", "{住处短句或已说过}，" + HOME_TEMP_PART.format(area="{区}")))
    rows.append(("一次超过 2 项", "只答前 2 项", TOO_MANY_ANSWER))
    return rows
