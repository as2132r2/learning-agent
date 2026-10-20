# 使用包含Python 3.12的精简Linux基础镜像。
FROM python:3.12-slim

# 让Python日志立即输出，并且不生成.pyc缓存文件。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HTTP_HOST=0.0.0.0 \
    HTTP_PORT=8000

WORKDIR /app

# 先复制依赖清单。代码变化时可以继续复用依赖安装缓存层。
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --requirement /app/requirements.txt

# 只复制Day13运行需要的Python代码。
COPY python/ /app/python/

# 创建数据库目录和非root运行用户。
RUN mkdir -p /app/database \
    && groupadd --system appgroup \
    && useradd --system --gid appgroup --home-dir /app appuser \
    && chown -R appuser:appgroup /app/database

USER appuser

EXPOSE 8000

# 数据库目录可以绑定到Docker volume，容器重建后仍保留会话。
VOLUME ["/app/database"]

# 不依赖curl，直接用Python标准库执行健康检查。
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "python/day13_deploy_api.py"]
