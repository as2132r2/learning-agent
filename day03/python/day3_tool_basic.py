from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""

    result = a * b

    return result


print("工具对象：")
print(multiply)

print("\n--------------------")

print("工具的 Python 类型：")
print(type(multiply))

print("\n--------------------")

print("工具名称：")
print(multiply.name)

print("\n--------------------")

print("工具说明：")
print(multiply.description)

print("\n--------------------")

print("工具参数：")
print(multiply.args)

print("\n--------------------")

print("工具参数的 JSON Schema：")
print(
    multiply.args_schema.model_json_schema()
)

print("\n--------------------")

result = multiply.invoke({
    "a": 23,
    "b": 47
})

print("工具执行结果：")
print(result)

print("\n结果的数据类型：")
print(type(result))
