from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpInteger, PULP_CBC_CMD

def max_feasible_sales(
    original_sales,
    min_dos_targets,
    max_dos_targets,
    wholesales,
    dealer_stock,
    frozen_months,
    selling_days
):
    """
    Computes the maximum feasible total sales over 12 months under hard inventory and
    days-of-supply constraints, honoring frozen months.

    Parameters
    ----------
    original_sales : list of int
        The initial sales plan for each month (length = 12).
    min_dos_targets : list of float
        Minimum acceptable days of supply at the end of each month.
    max_dos_targets : list of float
        Maximum acceptable days of supply at the end of each month.
    wholesales : list of int
        Number of vehicles entering inventory each month.
    dealer_stock : list of int
        Dealer stock at the end of each month (length = 12).
    frozen_months : list of int (0 or 1)
        Indicates if the month's sales are frozen (1 = fixed).
    selling_days : list of int
        Number of selling days in each month.

    Returns
    -------
    dict
        Dictionary containing:
        - "max_total_sales": float
        - "final_sales": list of optimized sales
        - "inventory_levels": list of month-end inventories
        - "days_of_supply": list of DoS per month
    """

    required_months = 12
    if not (
        len(original_sales)
        == len(min_dos_targets)
        == len(max_dos_targets)
        == len(wholesales)
        == len(dealer_stock)
        == len(frozen_months)
        == len(selling_days)
        == required_months
    ):
        raise ValueError("All input lists must have exactly 12 months of data.")

    # Determine the first adjustable month
    start_month = next((i for i, f in enumerate(frozen_months) if f == 0), required_months)

    # Initial inventory
    if start_month > 0:
        initial_inventory = dealer_stock[start_month - 1]
    else:
        initial_inventory = dealer_stock[0] - wholesales[0] + original_sales[0]

    # Initialize model
    model = LpProblem("max_feasible_sales", LpMaximize)

    adjusted_sales = [
        LpVariable(f"adjusted_sales_{i}", lowBound=0, cat=LpInteger)
        for i in range(start_month, required_months)
    ]
    inventory = [
        LpVariable(f"inventory_{i}", lowBound=0)
        for i in range(start_month, required_months)
    ]

    # Objective: maximize total sales
    model += lpSum(adjusted_sales)

    # Constraints
    for i in range(start_month, required_months):
        idx = i - start_month

        if frozen_months[i] == 1:
            model += adjusted_sales[idx] == original_sales[i]
        else:
            # no bounds needed, we let the model freely maximize within inventory constraints
            pass

        # Inventory balance
        if i == start_month:
            model += inventory[idx] == initial_inventory + wholesales[i] - adjusted_sales[idx]
        else:
            model += inventory[idx] == inventory[idx - 1] + wholesales[i] - adjusted_sales[idx]

        # DoS constraints
        model += inventory[idx] * selling_days[i] >= min_dos_targets[i] * adjusted_sales[idx]
        model += inventory[idx] * selling_days[i] <= max_dos_targets[i] * adjusted_sales[idx]

    # Solve
    solver = PULP_CBC_CMD(msg=True, gapRel=0.001, timeLimit=10)
    model.solve(solver)

    # Retrieve results
    final_sales = [
        original_sales[i] if frozen_months[i] == 1 else adjusted_sales[i - start_month].varValue
        for i in range(required_months)
    ]
    final_inventory = [
        dealer_stock[i] if i < start_month else inventory[i - start_month].varValue
        for i in range(required_months)
    ]
    monthly_dos = [
        final_inventory[i] / (final_sales[i] / selling_days[i])
        if final_sales[i] > 0 else None
        for i in range(required_months)
    ]

    return {
        "max_total_sales": sum(final_sales),
        "final_sales": final_sales,
        "inventory_levels": final_inventory,
        "days_of_supply": monthly_dos
    }
