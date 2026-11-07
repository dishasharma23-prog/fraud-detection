"""Browser tests (Playwright) for the static demo page and the page served by the API."""

import functools
import json
import re
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DEMO = json.loads((DOCS / "data.json").read_text())


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def demo_url():
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(DOCS))
    server = ThreadingHTTPServer(("127.0.0.1", free_port()), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


@pytest.fixture(scope="module")
def api_url():
    from app.main import app
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    page = browser.new_page()
    page.route(re.compile(r"fonts\.(googleapis|gstatic)\.com"), lambda route: route.abort())  # no network needed
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    yield page
    assert not errors, errors
    page.close()


def test_demo_shows_the_real_metrics(page, demo_url):
    page.goto(demo_url)
    m = DEMO["metrics"]
    expect(page.locator("#metrics")).to_contain_text(f"{m['pr_auc']:.3f}")
    expect(page.locator("#metrics")).to_contain_text(f"{m['roc_auc']:.3f}")
    expect(page.locator("#metrics")).to_contain_text(f"{m['rows']['test']:,}")


def test_demo_lists_every_transaction_and_filters(page, demo_url):
    page.goto(demo_url)
    items = page.locator("#list .item")
    expect(items).to_have_count(len(DEMO["transactions"]))
    fraud = sum(t["is_fraud"] for t in DEMO["transactions"])

    page.click('.filters button[data-f="1"]')
    expect(items).to_have_count(fraud)
    expect(page.locator("#list .tag.legit")).to_have_count(0)

    page.click('.filters button[data-f="0"]')
    expect(items).to_have_count(len(DEMO["transactions"]) - fraud)
    expect(page.locator("#list .tag.fraud")).to_have_count(0)


def test_selecting_a_transaction_shows_its_score_and_reasons(page, demo_url):
    page.goto(demo_url)
    for t in DEMO["transactions"][:5]:
        page.click(f'#list .item[data-id="{t["id"]}"]')
        r = t["result"]
        expect(page.locator("#title")).to_have_text(f"Transaction #{t['id']}")
        expect(page.locator("#prob")).to_have_text(f"{r['fraud_probability'] * 100:.1f}%")
        expect(page.locator("#pill")).to_have_text("Flag for review" if r["flagged"] else "Approve")
        expect(page.locator("#reasons .reason")).to_have_count(len(r["reasons"]))
        expect(page.locator("#actual")).to_contain_text("fraud" if t["is_fraud"] else "legit")


def test_demo_fits_a_phone_screen(browser, demo_url):
    page = browser.new_page(viewport={"width": 375, "height": 812})
    page.route(re.compile(r"fonts\.(googleapis|gstatic)\.com"), lambda route: route.abort())
    page.goto(demo_url)
    expect(page.locator("#list .item").first).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth") <= 375
    page.close()


def test_api_page_rescores_an_edited_transaction(page, api_url):
    page.goto(api_url)
    expect(page.locator("#prob")).not_to_be_empty()
    page.fill('[data-key="TransactionAmt"]', "4999")
    page.click("#score")
    expect(page.locator("#actual")).to_have_text("Edited transaction")
    expect(page.locator("#prob")).to_have_text(re.compile(r"^\d+\.\d%$"))


def test_api_page_explains_invalid_input(page, api_url):
    page.goto(api_url)
    expect(page.locator("#prob")).not_to_be_empty()
    page.fill('[data-key="TransactionAmt"]', "")
    page.click("#score")
    expect(page.locator("#hint")).to_contain_text("must include TransactionAmt")
