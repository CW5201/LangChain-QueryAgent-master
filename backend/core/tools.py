"""
SmartKB 工具定义模块

本模块定义了Agent可以使用的外部工具。

什么是工具？
    工具就是Agent可以调用的外部函数，类似于给AI装上了"手"，
    让它能够执行具体操作，而不只是"说话"。

可用工具：
┌─────────────────┬────────────────────────────────────┐
│     工具名称     │              功能说明               │
├─────────────────┼────────────────────────────────────┤
│ database_query  │ 执行SQL查询，获取数据库数据         │
│ web_search      │ 网络搜索，获取外部实时信息           │
└─────────────────┴────────────────────────────────────┘

工具设计原则：
1. 每个工具是一个独立的Python函数
2. 使用 @tool 装饰器注册到LangChain
3. 函数docstring要清晰描述工具功能（LLM会读取这个描述来决定是否使用工具）
"""

import os
import sqlite3
from langchain_core.tools import tool    # LangChain的工具装饰器


# ============================================================================
# 配置常量
# ============================================================================

# 示例数据库路径（可通过环境变量覆盖）
DB_PATH = os.getenv("SQLITE_DB_PATH", "data/sample.db")


# ============================================================================
# 数据库初始化
# ============================================================================

def init_sample_database():
    """
    初始化示例数据库
    
    创建两个示例表：
    1. sales - 销售数据表（产品线、年月、金额、数量、区域）
    2. users - 用户信息表（姓名、部门、职位、入职日期）
    
    插入的示例数据：
    - 3条产品线 × 6个月 = 18条销售记录
    - 5个示例员工
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # ---- 创建销售数据表 ----
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_line TEXT NOT NULL,   -- 产品线名称（电子产品/办公用品/食品饮料）
            year INTEGER NOT NULL,        -- 年份
            month INTEGER NOT NULL,       -- 月份
            amount REAL NOT NULL,         -- 销售金额（元）
            quantity INTEGER NOT NULL,    -- 销售数量
            region TEXT                   -- 销售区域（华东/华北/华南/华中/西南）
        )
    ''')
    
    # ---- 创建用户信息表 ----
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,           -- 姓名
            department TEXT,              -- 部门
            position TEXT,                -- 职位
            hire_date DATE                -- 入职日期
        )
    ''')
    
    # ---- 插入示例销售数据 ----
    sample_sales = [
        # 电子产品线（2023-2024年，华东/华北区域）
        ('电子产品', 2024, 1, 150000, 50, '华东'),
        ('电子产品', 2024, 2, 180000, 60, '华东'),
        ('电子产品', 2024, 3, 200000, 70, '华北'),
        ('电子产品', 2023, 1, 120000, 40, '华东'),
        ('电子产品', 2023, 2, 140000, 45, '华东'),
        ('电子产品', 2023, 3, 160000, 55, '华北'),
        # 办公用品线（2023-2024年，华南/华东区域）
        ('办公用品', 2024, 1, 50000, 100, '华南'),
        ('办公用品', 2024, 2, 55000, 110, '华南'),
        ('办公用品', 2024, 3, 60000, 120, '华东'),
        ('办公用品', 2023, 1, 40000, 80, '华南'),
        ('办公用品', 2023, 2, 45000, 90, '华南'),
        ('办公用品', 2023, 3, 48000, 95, '华东'),
        # 食品饮料线（2023-2024年，华中/西南区域）
        ('食品饮料', 2024, 1, 80000, 200, '华中'),
        ('食品饮料', 2024, 2, 90000, 220, '华中'),
        ('食品饮料', 2024, 3, 95000, 230, '西南'),
        ('食品饮料', 2023, 1, 70000, 175, '华中'),
        ('食品饮料', 2023, 2, 75000, 185, '华中'),
        ('食品饮料', 2023, 3, 80000, 200, '西南'),
    ]
    
    # 批量插入（IGNORE避免重复插入）
    cursor.executemany('''
        INSERT OR IGNORE INTO sales (product_line, year, month, amount, quantity, region)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', sample_sales)
    
    # ---- 插入示例用户数据 ----
    sample_users = [
        ('张三', '技术部', '高级工程师', '2020-03-15'),
        ('李四', '市场部', '市场经理', '2019-07-20'),
        ('王五', '财务部', '财务主管', '2018-11-10'),
        ('赵六', '技术部', '技术总监', '2017-05-01'),
        ('钱七', '人事部', '人事经理', '2020-09-25'),
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO users (name, department, position, hire_date)
        VALUES (?, ?, ?, ?)
    ''', sample_users)
    
    conn.commit()
    conn.close()


# ============================================================================
# 工具定义
# ============================================================================

@tool
def database_query(sql: str) -> str:
    """
    执行SQL查询并返回结果（仅支持SELECT查询）

    这是一个数据库查询工具，可以查询示例数据库中的销售数据和用户数据。
    支持标准SQL语法，包括SELECT、聚合函数（SUM/COUNT/AVG）、JOIN等。

    当用户问到数据相关的问题时（如"查询销售总额"、"有多少员工"），
    Agent会自动调用这个工具。

    Args:
        sql: 要执行的SQL查询语句（仅限SELECT）

    Returns:
        查询结果的JSON字符串格式

    Examples:
        >>> database_query("SELECT * FROM sales WHERE year = 2024")
        >>> database_query("SELECT product_line, SUM(amount) as total FROM sales GROUP BY product_line")
        >>> database_query("SELECT * FROM users WHERE department = '技术部'")
    """
    try:
        # 确保安全：只允许 SELECT 语句
        cleaned = sql.strip().upper()
        if not cleaned.startswith("SELECT"):
            return "错误：仅支持SELECT查询，不允许执行写入或删除操作"

        # 确保示例数据库已初始化
        if not os.path.exists(DB_PATH):
            init_sample_database()

        # 连接数据库并执行查询
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)

        # 获取列名（用于构建字典）
        column_names = [desc[0] for desc in cursor.description] if cursor.description else []

        # 获取查询结果
        results = cursor.fetchall()
        conn.close()

        # 处理空结果
        if not results:
            return "查询结果为空"

        # 将结果转换为字典列表（更易读）
        result_list = []
        for row in results:
            row_dict = {col: row[i] for i, col in enumerate(column_names)}
            result_list.append(row_dict)

        return str(result_list)

    except Exception as e:
        return f"查询执行错误: {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    使用DuckDuckGo搜索网络信息
    
    搜索互联网获取最新信息，返回前3条相关结果。
    适用于需要实时信息或外部知识的问题。
    
    当用户问到实时信息（如"今天天气"、"最新新闻"）时，
    Agent会自动调用这个工具。
    
    Args:
        query: 搜索关键词
        
    Returns:
        搜索结果摘要，包含标题和内容
        
    Examples:
        >>> web_search("Python最新版本")
        >>> web_search("成都天气预报")
        >>> web_search("2024年AI发展趋势")
    """
    try:
        from duckduckgo_search import DDGS
        
        # 执行搜索，限制返回3条结果
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
            if not results:
                return "未找到相关搜索结果"
            
            # 格式化搜索结果
            formatted = []
            for i, result in enumerate(results, 1):
                title = result.get('title', '无标题')
                body = result.get('body', '无摘要')
                formatted.append(f"{i}. {title}\n{body}")
            
            return "\n\n".join(formatted)
            
    except ImportError:
        return "搜索功能需要安装duckduckgo_search库: pip install duckduckgo-search"
    except Exception as e:
        return f"搜索执行错误: {str(e)}"


# ============================================================================
# 工具注册
# ============================================================================

# 所有可用工具列表（Agent会自动加载这些工具）
TOOLS = [database_query, web_search]


def get_tools():
    """
    获取所有可用工具
    
    Returns:
        工具函数列表，供AgentExecutor使用
    """
    return TOOLS
