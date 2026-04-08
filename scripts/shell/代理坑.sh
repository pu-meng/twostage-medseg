!/bin/bash
============================================================
Claude Code 无GUI服务器配置完整指南
作者:pumengyu @ 山东大学2080服务器
日期:2026-03-24
耗时:一上午(踩了所有坑)
============================================================

============================================================
【背景】
服务器:Ubuntu 无GUI,SSH连接
目标:在服务器上使用 Claude Code(Pro账号)
难点:服务器在国内,api.anthropic.com 被封,需要代理
============================================================


============================================================
第一部分:排查代理问题
============================================================


⚠️ 坑2:Clash API 有 secret 时需要加 Bearer token
先查 secret:
  sudo grep -E "secret|external-controller" /srv/clash/config.yaml
如果有 secret:
  curl -s --noproxy '*' -H "Authorization: Bearer 你的secret" http://127.0.0.1:9090/proxies
如果没有 secret(本案例):直接访问即可

⚠️ 坑3:节点 "alive: false" 但面板显示已选中
查看节点状态(重要！):
  curl -s --noproxy '*' http://127.0.0.1:9090/proxies/AI | python3 -m json.tool
  
看 "alive" 字段,false = 节点已挂,换节点没用,要换机场

⚠️ 坑4:TLS reset 不是 Clash 的问题,是节点的问题
现象:curl 返回 "Recv failure: Connection reset by peer"
原因:节点出口被封,TLS握手失败
解决:换节点或换机场

批量测试节点是否可用(重要工具):


test_nodes() {
    for i in 01 02 03 04 05 06 07 08 09 10; do
        node="🇺🇸 美国 $i"
        curl -s --noproxy '*' -X PUT "http://127.0.0.1:9090/proxies/AI" \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"$node\"}" > /dev/null 2>&1
        sleep 0.5
        result=$(curl -x http://127.0.0.1:7890 -s --max-time 6 https://api.ipify.org 2>/dev/null)
        if [ -n "$result" ]; then
            echo "✅ $node → $result"
        else
            echo "❌ $node"
        fi
    done
}


============================================================
第二部分:机场全挂时的解决方案
============================================================

⚠️ 坑5:机场节点全挂(美国/日本/新加坡/台湾全部❌)
原因:机场质量差,节点被批量封
解决:换机场订阅,推荐七七VPN(qi77.cc)
  - 去官网登录 → 下载 → Clash订阅文件
  - 节点信息在 yaml 文件里,包含服务器地址+密码

⚠️ 坑6:订阅链接直连下载失败(代理本身坏了无法用代理下载)
错误:curl: (35) Recv failure: Connection reset by peer
解决:在本地电脑浏览器下载,scp传到服务器
  本地PowerShell:scp C:\Users\用户名\Downloads\配置.yaml pumengyu@服务器IP:/home/pumengyu/

⚠️ 坑7:七七VPN的Linux客户端(.deb)在无GUI服务器上跑不了
错误:Missing X server or $DISPLAY / Segmentation fault
原因:七七客户端是图形界面程序,无GUI服务器没有显示器
解决:不用七七客户端,只用它的节点信息 + Clash运行


============================================================
第三部分:自己跑一个 Clash 实例(用七七节点)
============================================================

从七七VPN订阅里提取节点,写成 Clash 配置
⚠️ 坑8:需要 Country.mmdb 地理数据库,否则 Clash 启动失败
错误:can't initial MMDB: can't download MMDB
解决:从实验室已有的 Clash 复制
  sudo cp /srv/clash/Country.mmdb /home/pumengyu/Country.mmdb

setup_qi77_clash() {
    # 写配置文件(注意用自己账号的节点信息)
    cat > /home/pumengyu/qi77_clash.yaml << 'EOF'
mixed-port: 7891
allow-lan: false
mode: global
log-level: info
external-controller: 127.0.0.1:9091

proxies:
  - name: 美国01-GPT
    type: ss
    server: mg1b.srooiar.top
    port: 38258
    cipher: aes-128-gcm
    password: "你的密码"
  - name: 美国02-GPT
    type: ss
    server: mg2.srooiar.top
    port: 58508
    cipher: aes-128-gcm
    password: "你的密码"
  - name: 新加坡04-GPT
    type: ss
    server: xjp4.xvorep.top
    port: 26176
    cipher: aes-128-gcm
    password: "你的密码"

proxy-groups:
  - name: Proxy
    type: select
    proxies:
      - 美国01-GPT
      - 美国02-GPT
      - 新加坡04-GPT

rules:
  - MATCH,Proxy
EOF

    echo "配置文件已写入"
}

⚠️ 坑9:端口 7891 被之前的进程占用
错误:listen tcp 127.0.0.1:7891: bind: address already in use
解决:
  sudo fuser -k 7891/tcp 7891/udp
  pkill -f qi77_clash

⚠️ 坑10:Clash API(9091端口)的 GLOBAL 组默认是 DIRECT
启动后必须手动切换到 Proxy:



switch_to_proxy() {
    curl -s --noproxy '*' -X PUT http://127.0.0.1:9091/proxies/GLOBAL \
        -H "Content-Type: application/json" \
        -d '{"name":"Proxy"}'
    echo "已切换到 Proxy"
}

# 启动七七Clash实例

start_qi77() {
    # 先检查是否已在运行
    if pgrep -f "qi77_clash.yaml" > /dev/null 2>&1; then
        echo "qi77 clash 已在运行"
        return
    fi
    sudo fuser -k 7891/tcp 7891/udp 2>/dev/null
    sleep 1
    /usr/bin/clash -d /home/pumengyu/ -f /home/pumengyu/qi77_clash.yaml \
        > /home/pumengyu/clash.log 2>&1 &
    sleep 3
    echo "启动完成,测试连接:"
    curl -x http://127.0.0.1:7891 -s --max-time 10 https://api.ipify.org && echo ""
}


============================================================
第四部分:Claude Code 登录
============================================================

⚠️ 坑11:claude auth login 需要浏览器(无GUI服务器没有)
解决:先设置代理,login会输出一个URL,复制到本地浏览器打开
  export HTTPS_PROXY=http://127.0.0.1:7891
  claude auth login
  复制终端里的URL到本地浏览器,登录Pro账号,自动完成认证

⚠️ 坑12:登录成功但 claude 启动报 ConnectionRefused
原因:export 的环境变量没有传递给 claude 进程
错误方式(不起作用):
  export HTTPS_PROXY=... 然后在另一个终端运行 claude
✅ 正确方式(在同一个终端,同一行或先export再运行):
  export HTTPS_PROXY=http://127.0.0.1:7891
  export HTTP_PROXY=http://127.0.0.1:7891
  export https_proxy=http://127.0.0.1:7891   # 注意小写也要设！
  export http_proxy=http://127.0.0.1:7891
  claude

⚠️ 坑13:只设置大写 HTTPS_PROXY 不够,必须大小写都设
原因:Node.js 程序(Claude Code是Node写的)同时读大小写两种变量
✅ 必须同时设置四个:


set_proxy() {
    export HTTPS_PROXY=http://127.0.0.1:7891
    export HTTP_PROXY=http://127.0.0.1:7891
    export https_proxy=http://127.0.0.1:7891
    export http_proxy=http://127.0.0.1:7891
    echo "代理已设置"
}

⚠️ 坑14:claude --proxy 参数不存在
错误:error: unknown option '--proxy'
Claude Code 不支持命令行代理参数,只能用环境变量

⚠️ 坑15:claude config set httpProxy 也没用
Claude Code 的 config 里 httpProxy 不生效(v2.1.81版本)
唯一有效方式:环境变量


============================================================
第五部分:验证命令合集
============================================================

# 验证代理是否走通

verify_proxy() {
    echo "=== 出口IP ==="
    curl -x http://127.0.0.1:7891 -s --max-time 10 https://ipinfo.io/json | python3 -m json.tool
    
    echo "=== 验证 Anthropic 可达 ==="
    curl -x http://127.0.0.1:7891 -s --max-time 10 \
        https://api.anthropic.com/v1/models \
        -H "anthropic-version: 2023-06-01" \
        -H "x-api-key: test" | head -c 100
    echo ""
    # 返回 {"type":"error"...authentication_error} = 网络通了(401认证错误是正常的)
    # 返回空或超时 = 节点不通
}


============================================================
第六部分:写入 .bashrc 的正确姿势
============================================================

⚠️ 坑16:.bashrc 里 clash 启动了两次导致端口冲突
原因:手动添加时不小心写了两遍
解决:用 pgrep 判断是否已启动,避免重复

# 以下是写入 .bashrc 末尾的正确内容:
cat << 'BASHRC_EOF'
# ===== qi77 代理配置 =====
# 只在未运行时启动(避免重复)
if ! pgrep -f "qi77_clash.yaml" > /dev/null 2>&1; then
    /usr/bin/clash -d /home/pumengyu/ -f /home/pumengyu/qi77_clash.yaml \
        > /home/pumengyu/clash.log 2>&1 &
fi
# 大小写都要设,Node.js 程序需要
export HTTPS_PROXY=http://127.0.0.1:7891
export HTTP_PROXY=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export http_proxy=http://127.0.0.1:7891
# ===== 代理配置结束 =====
BASHRC_EOF


============================================================
第七部分:快速启动 Claude Code(每次使用)
============================================================

确认 clash 在跑:
  ps aux | grep qi77_clash | grep -v grep

确认代理可用:
  curl -x http://127.0.0.1:7891 -s --max-time 5 https://api.ipify.org

启动 Claude Code:
  claude

如果 .bashrc 配置正确,每次 SSH 登录后直接运行 claude 即可


============================================================
总结:问题根本原因链
============================================================
1. 实验室机场(AmyTelecom)节点全挂 → 无法直接用 Clash
2. 无GUI服务器无法运行七七图形客户端 → 改用七七节点+Clash
3. Clash API 被代理本身拦截 → 加 --noproxy '*'
4. Claude Code 不认环境变量 → 大小写都要设,同一终端
5. .bashrc 重复启动 → 加 pgrep 判断
============================================================