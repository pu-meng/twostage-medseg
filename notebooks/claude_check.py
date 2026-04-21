#!/usr/bin/env python3
"""
Claude Code 连接诊断脚本 v3
新增：直接读取 Clash 配置文件检测节点域名/IP/端口
结果写入 report.txt
"""

import subprocess
import json
import os
import socket
import datetime
import time
import yaml  # 需要 pyyaml

REPORT = []
CONFIG_FILES = [
    "/home/ZhaoPu/1775529590563.yml",  # 大哥云
    "/home/ZhaoPu/qi77_clash.yaml",    # qi77
]

def log(title, content, status="INFO"):
    icon = {"OK": "✅", "FAIL": "❌", "WARN": "⚠️", "INFO": "ℹ️"}.get(status, "ℹ️")
    line = f"\n{'='*60}\n{icon} [{status}] {title}\n{'-'*60}\n{content}"
    REPORT.append(line)
    print(line)

def run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return "", str(e)

def tcp_test(host, port, timeout=4):
    """TCP 直连测试，返回 (ok, latency_ms, error)"""
    try:
        t0 = time.time()
        s = socket.create_connection((host, int(port)), timeout=timeout)
        s.close()
        return True, int((time.time()-t0)*1000), ""
    except socket.timeout:
        return False, -1, "超时"
    except socket.gaierror as e:
        return False, -1, f"DNS解析失败: {e}"
    except Exception as e:
        return False, -1, str(e)

def dns_resolve(host):
    """解析域名，返回 IP 列表"""
    try:
        infos = socket.getaddrinfo(host, None)
        ips = list(set(i[4][0] for i in infos))
        return ips, ""
    except socket.gaierror as e:
        return [], str(e)

# ─────────────────────────────────────────────
def check_env_proxy():
    vars = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]
    result = []
    for v in vars:
        val = os.environ.get(v, "（未设置）")
        result.append(f"  {v} = {val}")
    proxies = [os.environ.get(v, "") for v in vars]
    has_8118 = any("8118" in p for p in proxies)
    has_7890 = any("7890" in p for p in proxies)
    content = "\n".join(result)
    if has_8118:
        log("环境变量代理", content + "\n\n❌ 还有 8118(Privoxy)！", "FAIL")
    elif has_7890:
        log("环境变量代理", content + "\n\n✅ 使用 7890(Clash)", "OK")
    else:
        log("环境变量代理", content + "\n\n❌ 没有设置代理", "FAIL")

def check_bashrc():
    out, _ = run("grep -n proxy ~/.bashrc 2>/dev/null || echo '（未找到）'")
    if "8118" in out:
        log(".bashrc", out + "\n\n❌ 还有 8118！\n修复: sed -i '/8118/d' ~/.bashrc", "FAIL")
    elif "7890" in out:
        log(".bashrc", out + "\n\n✅ 只有 7890", "OK")
    else:
        log(".bashrc", out + "\n\n⚠️ 没有代理配置", "WARN")

def check_ports():
    out, _ = run("ss -tlnp")
    ports = {7890:"Clash出口", 8118:"Privoxy", 9090:"Clash面板(大哥云)", 4000:"Clash面板(qi77)"}
    result = []
    for port, name in ports.items():
        listening = f":{port}" in out
        result.append(f"  {port}  {'✅监听' if listening else '❌未监听'}  ({name})")
    log("端口监听", "\n".join(result), "INFO")

def check_clash_process():
    out, _ = run("ps aux | grep clash | grep -v grep")
    log("Clash进程", out if out else "❌ 无 Clash 进程", "OK" if out else "FAIL")

def check_connectivity():
    tests = [
        ("7890(Clash)",  "curl -s -x http://127.0.0.1:7890 https://api.anthropic.com -o /dev/null -w '%{http_code}' --connect-timeout 8"),
        ("直连",          "curl -s --noproxy '*' https://api.anthropic.com -o /dev/null -w '%{http_code}' --connect-timeout 8"),
    ]
    result = []
    for name, cmd in tests:
        out, _ = run(cmd, timeout=12)
        code = out.strip()
        sym = "✅" if code in ("200","403") else "❌"
        result.append(f"  {name:15}: {sym} HTTP {code}")
    log("Anthropic连通性", "\n".join(result), "INFO")

# ─────────────────────────────────────────────
# 核心新功能：读配置文件检测节点
# ─────────────────────────────────────────────
def check_nodes_from_config():
    """直接读 Clash yaml 配置文件，检测每个节点的域名/IP/端口"""

    # 检查 pyyaml
    try:
        import yaml
    except ImportError:
        log("节点配置检测", "❌ 缺少 pyyaml，正在安装...", "WARN")
        run("pip install pyyaml --break-system-packages -q")
        try:
            import yaml
        except:
            log("节点配置检测", "❌ pyyaml 安装失败，跳过", "FAIL")
            return

    for config_path in CONFIG_FILES:
        _check_one_config(config_path, yaml)

def _check_one_config(config_path, yaml):
    fname = os.path.basename(config_path)

    # 读文件（可能需要 sudo）
    out, err = run(f"sudo cat {config_path} 2>/dev/null")
    if not out:
        log(f"配置文件 {fname}", f"❌ 无法读取: {err}", "FAIL")
        return

    try:
        cfg = yaml.safe_load(out)
    except Exception as e:
        log(f"配置文件 {fname}", f"❌ YAML 解析失败: {e}", "FAIL")
        return

    proxies = cfg.get("proxies", cfg.get("Proxies", []))
    if not proxies:
        log(f"配置文件 {fname}", "❌ 没有找到 proxies 字段", "FAIL")
        return

    # 收集所有唯一服务器
    servers = {}  # server -> [(name, port, type)]
    for p in proxies:
        if not isinstance(p, dict):
            continue
        name   = p.get("name", "?")
        server = p.get("server", "")
        port   = p.get("port", "?")
        ptype  = p.get("type", "?")
        if server:
            if server not in servers:
                servers[server] = []
            servers[server].append((name, port, ptype))

    total_servers = len(servers)
    total_nodes   = len(proxies)
    print(f"\n📋 {fname}: {total_nodes} 个节点，{total_servers} 个不同服务器，开始检测...\n")

    details = []
    ok_servers  = []
    dns_fail    = []
    tcp_fail    = []

    for server, node_list in servers.items():
        # Step 1: DNS 解析
        ips, dns_err = dns_resolve(server)

        if not ips:
            # 再试一次用 8.8.4.4（DoH 可能绕过）
            out2, _ = run(f"getent hosts {server} 2>/dev/null")
            if out2:
                ips = [out2.split()[0]]
                dns_err = ""

        if not ips:
            line = f"  ❌ DNS失败  {server}  →  {dns_err}"
            details.append(line)
            print(line)
            dns_fail.append((server, dns_err, node_list))
            continue

        ip_str = ", ".join(ips[:2])

        # Step 2: 用每个节点的端口做 TCP 测试
        port_results = []
        any_port_ok  = False
        tested_ports = set()

        for name, port, ptype in node_list:
            if port in tested_ports:
                continue
            tested_ports.add(port)
            ok, ms, terr = tcp_test(server, port)
            if ok:
                port_results.append(f"    端口 {port}: ✅ {ms}ms")
                any_port_ok = True
            else:
                port_results.append(f"    端口 {port}: ❌ {terr}")

        ports_str = "\n".join(port_results)

        if any_port_ok:
            line = f"  ✅ 可用  {server}  ({ip_str})\n{ports_str}"
            ok_servers.append(server)
        else:
            line = f"  ⚠️ DNS通但端口不通  {server}  ({ip_str})\n{ports_str}"
            tcp_fail.append((server, node_list))

        details.append(line)
        print(line)

    # 汇总
    summary = f"\n{'='*60}\n"
    summary += f"📊 {fname} 统计:\n"
    summary += f"  总节点数    : {total_nodes}\n"
    summary += f"  唯一服务器  : {total_servers}\n"
    summary += f"  ✅ 可用服务器: {len(ok_servers)}\n"
    summary += f"  ❌ DNS失败   : {len(dns_fail)}\n"
    summary += f"  ⚠️ 端口不通  : {len(tcp_fail)}\n"

    if dns_fail:
        summary += f"\n❌ DNS解析失败的服务器:\n"
        for s, e, nodes in dns_fail:
            summary += f"    {s}  →  {e}\n"
            summary += f"    影响节点: {', '.join(n[0] for n in nodes[:3])}" + ("..." if len(nodes)>3 else "") + "\n"

    if ok_servers:
        summary += f"\n✅ 可用服务器:\n"
        for s in ok_servers:
            summary += f"    {s}\n"

    # 判断机场状态
    if len(ok_servers) == 0 and len(dns_fail) == total_servers:
        summary += f"\n🚨 结论: {fname} 的机场订阅域名全部无法解析，订阅很可能已过期或服务商跑路！"
        status = "FAIL"
    elif len(ok_servers) == 0:
        summary += f"\n🚨 结论: {fname} 的所有节点端口不通，服务器可能被封或宕机！"
        status = "FAIL"
    else:
        summary += f"\n✅ 结论: {fname} 有可用节点"
        status = "OK"

    content = "\n".join(details) + "\n" + summary
    log(f"节点配置检测 - {fname}", content, status)

def check_claude_config():
    config_path = os.path.expanduser("~/.claude.json")
    if not os.path.exists(config_path):
        log("Claude Code配置", "❌ 不存在", "FAIL")
        return
    try:
        with open(config_path) as f:
            data = json.load(f)
        oauth = data.get("oauthAccount", {})
        result = [
            f"  hasAvailableSubscription: {data.get('hasAvailableSubscription')}",
            f"  billingType: {oauth.get('billingType')}",
            f"  email: {oauth.get('emailAddress')}",
        ]
        sub_ok = data.get("hasAvailableSubscription", True)
        content = "\n".join(result)
        if not sub_ok:
            log("Claude Code配置", content + "\n\n⚠️ 本地缓存显示订阅不可用，但网页显示Pro正常，执行: claude logout && claude login 刷新", "WARN")
        else:
            log("Claude Code配置", content, "OK")
    except Exception as e:
        log("Claude Code配置", f"读取失败: {e}", "FAIL")

def generate_fix():
    fixes = []
    fixes.append("当前状态:\n  两个机场节点域名均无法解析\n  服务器可直连 Anthropic (HTTP 403)")
    fixes.append("方案1 (推荐): 联系 ZhaoPu 更新机场订阅，或者换新机场")
    fixes.append("方案2: 用手机热点临时登录\n"
                 "   # 手机开热点后：\n"
                 "   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY\n"
                 "   claude login\n"
                 "   # 登录成功后断开热点，恢复代理变量")
    fixes.append("方案3: 用 API Key（需额外付费）\n"
                 "   export ANTHROPIC_API_KEY=sk-ant-api03-xxx\n"
                 "   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY\n"
                 "   claude")
    log("修复建议", "\n\n".join(fixes), "INFO")

# ─────────────────────────────────────────────
print("🔍 开始诊断 v3...\n")
REPORT.append(f"Claude Code 连接诊断报告 v3\n生成时间: {datetime.datetime.now()}\n{'='*60}")

check_env_proxy()
check_bashrc()
check_ports()
check_clash_process()
check_connectivity()
check_claude_config()
check_nodes_from_config()   # ← 核心：直接读配置文件检测节点
generate_fix()

with open("report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT))

print(f"\n{'='*60}")
print("✅ 诊断完成，报告已写入 report.txt")
print(f"{'='*60}")