import json
import numpy as np


def bond_price(face_value, coupon_rate, discount_rate, years, payments_per_year=2):
    n_periods = years * payments_per_year
    coupon_payment = (coupon_rate / payments_per_year) * face_value
    discount_rate = discount_rate / payments_per_year / \
        100

    pv_coupons = sum(coupon_payment / (1 + discount_rate)
                     ** t for t in range(1, n_periods + 1))

    pv_face_value = face_value / (1 + discount_rate) ** n_periods

    return pv_coupons + pv_face_value


with open('Profit_Loss/actual_bond_yields.json', 'r') as file:
    bond_yields = json.load(file)

maturity = 10
face_value = 100

bond_prices = {}

for date, yields in bond_yields.items():
    bond_prices[date] = {}
    for bond_type, yield_rate in yields.items():
        coupon_rate = yield_rate
        discount_rate = yield_rate
        price = bond_price(face_value, coupon_rate, discount_rate, maturity)
        bond_prices[date][bond_type] = round(price, 2)

with open('bond_prices_actual.json', 'w') as file:
    json.dump(bond_prices, file, indent=4)

print("Bond prices calculated and saved to 'bond_prices.json'.")
