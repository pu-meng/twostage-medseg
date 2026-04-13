#!/bin/bash
# fix_claude.sh - 一键诊断并修复 Claude Code 代理
# 用法: bash ~/fix_claude.sh

echo "======================================"
echo "  Claude Code 代理诊断工具"
echo "======================================"

# ─────────────────────────────────────────
# 第1步:检查节点是否能连上 Anthropic
# ─────────────────────────────────────────
echo ""
echo "【第1步】检查节点 (socks5://127.0.0.1:7891)..."
echo "  原理:7891 是 root 的 clash，走美国节点出去"

RESULT=$(curl -x socks5://127.0.0.1:7891 https://api.anthropic.com \
  -o /dev/null -w "%{http_code}" --connect-timeout 8 2>/dev/null)

if [ "$RESULT" = "404" ]; then
  echo "  ✅ 节点正常 (返回404，说明Anthropic收到了请求)"
else
  echo "  ❌ 节点不通 (返回$RESULT，需要切换节点)"
  echo ""
  echo "  正在自动切换节点，请稍候..."
  
  # 获取所有节点名
  NODES=$(curl --noproxy '*' -s http://127.0.0.1:9090/proxies 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    for k,v in d['proxies'].items():
        if v.get('type') not in ('Selector','URLTest','Fallback','LoadBalance','ReURLTest') and k not in ('DIRECT','REJECT'):
            print(k)
except:
    pass
")

  FOUND=0
  while IFS= read -r node; do
    [ -z "$node" ] && continue
    curl --noproxy '*' -X PUT http://127.0.0.1:9090/proxies/GLOBAL \
      -H "Content-Type: application/json" \
      -d "{\"name\": \"$node\"}" > /dev/null 2>&1
    R=$(curl -x socks5://127.0.0.1:7891 https://api.anthropic.com \
      -o /dev/null -w "%{http_code}" --connect-timeout 8 2>/dev/null)
    echo "  试验节点: $node → $R"
    if [ "$R" = "404" ]; then
      echo "  ✅ 找到可用节点: $node"
      FOUND=1
      break
    fi
  done <<< "$NODES"

  if [ "$FOUND" = "0" ]; then
    echo ""
    echo "  ❌ 所有节点都不通，可能是机场故障，请联系赵璞或等待机场恢复"
    echo "  退出诊断"
    exit 1
  fi
fi

# ─────────────────────────────────────────
# 第2步:检查 privoxy
# ─────────────────────────────────────────
echo ""
echo "【第2步】检查 privoxy (socks5→http转换器)..."
echo "  原理:Claude Code(Node.js)不支持socks5，privoxy把7891转成http的8118"

PRIVOXY_OK=$(curl -x http://127.0.0.1:8118 https://api.anthropic.com \
  -o /dev/null -w "%{http_code}" --connect-timeout 8 2>/dev/null)

if [ "$PRIVOXY_OK" = "404" ]; then
  echo "  ✅ privoxy 正常 (8118端口可用)"
else
  echo "  ❌ privoxy 不通，正在修复..."
  
  # 检查是否安装
  if ! command -v privoxy &>/dev/null; then
    echo "  安装 privoxy..."
    sudo -E env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      apt-get install -y privoxy -q
  fi

  # 检查转发规则
  if ! grep -q "forward-socks5 / 127.0.0.1:7891" /etc/privoxy/config 2>/dev/null; then
    echo "  添加转发规则..."
    echo "forward-socks5 / 127.0.0.1:7891 ." | sudo tee -a /etc/privoxy/config > /dev/null
  fi

  sudo systemctl restart privoxy
  sleep 1

  PRIVOXY_OK2=$(curl -x http://127.0.0.1:8118 https://api.anthropic.com \
    -o /dev/null -w "%{http_code}" --connect-timeout 8 2>/dev/null)
  if [ "$PRIVOXY_OK2" = "404" ]; then
    echo "  ✅ privoxy 修复成功"
  else
    echo "  ❌ privoxy 修复失败，请手动检查: sudo systemctl status privoxy"
    exit 1
  fi
fi

# ─────────────────────────────────────────
# 第3步:检查环境变量
# ─────────────────────────────────────────
echo ""
echo "【第3步】检查环境变量..."
echo "  原理:程序通过这4个变量知道该把流量发给谁"

NEED_EXPORT=0
if [ "$HTTPS_PROXY" != "http://127.0.0.1:8118" ]; then
  echo "  ❌ 环境变量未设置，正在设置..."
  NEED_EXPORT=1
else
  echo "  ✅ 环境变量已设置 ($HTTPS_PROXY)"
fi

# ─────────────────────────────────────────
# 完成
# ─────────────────────────────────────────
echo ""
echo "======================================"
echo "  ✅ 诊断完成，执行以下命令后可用:"
echo "======================================"
echo ""
echo "  export http_proxy=http://127.0.0.1:8118"
echo "  export https_proxy=http://127.0.0.1:8118"
echo "  export HTTP_PROXY=http://127.0.0.1:8118"
echo "  export HTTPS_PROXY=http://127.0.0.1:8118"
echo ""
echo "  然后运行: claude"
echo ""
echo "  提示:把上面4行加到 ~/.bashrc 可以永久生效"