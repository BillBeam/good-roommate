"""agent 的长期记忆：键值笔记，跨会话保留。这是 agent 唯一的跨会话信息来源。"""

MAX_ENTRIES = 40
MAX_KEY = 40
MAX_VALUE = 300


class MemoryStore:
    def __init__(self):
        self.entries = {}  # key -> {"value": str, "week": int}

    def read(self):
        if not self.entries:
            return "（记忆为空）"
        return "\n".join(f"- {k}: {e['value']}（第 {e['week']} 周写入）" for k, e in self.entries.items())

    def write(self, key, value, week):
        key, value = str(key).strip()[:MAX_KEY], str(value).strip()[:MAX_VALUE]
        if not key:
            return "错误：key 不能为空"
        if key not in self.entries and len(self.entries) >= MAX_ENTRIES:
            return f"错误：记忆已满（最多 {MAX_ENTRIES} 条），请先删除或覆盖已有条目"
        self.entries[key] = {"value": value, "week": week}
        return "已保存"

    def delete(self, key):
        if self.entries.pop(str(key).strip(), None) is None:
            return "没有这条记忆"
        return "已删除"

    def state_dict(self):
        return {"entries": self.entries}

    def load_state_dict(self, d):
        self.entries = {k: dict(v) for k, v in d["entries"].items()}
