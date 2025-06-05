from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpInteger, PULP_CBC_CMD

def sales_repattern_optimize(
    original_sales,
    sales_target,
    min_dos_targets,
    max_dos_targets,
    wholesales,
    dealer_stock,
    frozen_months,
    selling_days,
    alpha=1.0,
    lambda_penalty=10.0,
):
    """
    Rebalances monthly sales to align with an annual sales target while preserving seasonal proportions
    and enforcing days of supply inventory constraints.

    The model adjusts monthly sales to minimize the spread between the highest and lowest ratios of
    adjusted sales to original planned sales (R_plus - R_minus), effectively preserving existing seasonality
    in the current sales plan as much as possible. It allows for a deviation from the annual sales target and penalizes that
    deviation in the objective function. Inventory levels are evaluated in terms of days of supply,
    which is calculated as ending inventory divided by sales per day (DSR), and bounded by user-specified
    minimum and maximum thresholds.

    Parameters
    ----------
    original_sales : list of int
        The initial sales plan for each month (length = 12).
    sales_target : int
        The total annual sales target to achieve (with slack allowed).
    min_dos_targets : list of float
        Minimum acceptable days of supply at the end of each month (length = 12).
    max_dos_targets : list of float
        Maximum acceptable days of supply at the end of each month (length = 12).
    wholesales : list of int
        Number of vehicles entering inventory each month (length = 12).
    dealer_stock : list of int
        Dealer stock at the end of each month.
    frozen_months : list of int (0 or 1)
        Indicator for whether sales are fixed (1 = frozen, 0 = adjustable) for each month (length = 12).
    selling_days : list of int
        Number of selling days in each month (length = 12).
    alpha : float, optional (default=1.0)
        Weight on the spread between R_plus and R_minus in the objective function.
    lambda_penalty : float, optional (default=10.0)
        Penalty weight on deviation from the annual sales target.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - "final_sales": list of optimized sales per month (integers represented as floats)
        - "R_plus": maximum adjusted/original ratio achieved
        - "R_minus": minimum adjusted/original ratio achieved
        - "ratios": list of ratios of adjusted sales to original sales
        - "target_deviation": amount the optimized total sales deviate from the input target
        - "inventory_levels": list of ending inventory levels for each month
        - "days_of_supply": DoS per month

    """
    # Validate we receive 12 months of data for relevant inputs
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

    # Identify first non-frozen month
    start_month = next(
        (i for i, frozen in enumerate(frozen_months) if frozen == 0), required_months
    )

    # Initial inventory from dealer stock
    if start_month > 0:
        initial_inventory = dealer_stock[start_month - 1]
    else:
        initial_inventory = dealer_stock[0] - wholesales[0] + original_sales[0]

    # Adjust target to reflect already actualized sales
    adjusted_sales_target = sales_target - sum(original_sales[:start_month])
    if adjusted_sales_target < 0:
        print("Warning: Historical sales exceed target. Adjusted target is negative.")
        adjusted_sales_target = 0

    # Initialize model
    model = LpProblem("sales_repattern", LpMinimize)

    # Decision variables (only from start_month onward)
    adjusted_sales = [
        LpVariable(f"adjusted_sales_{i}", lowBound=0, cat=LpInteger)
        for i in range(start_month, required_months)
    ]
    inventory = [
        LpVariable(f"inventory_{i}", lowBound=0)
        for i in range(start_month, required_months)
    ]
    R_plus = LpVariable("R_plus", lowBound=0)
    R_minus = LpVariable("R_minus", lowBound=0)
    sls_tgt_deviation = LpVariable("sls_tgt_deviation")
    sls_tgt_dev_abs = LpVariable("sls_tgt_dev_abs", lowBound=0)

    # Objective
    model += alpha * (R_plus - R_minus) + lambda_penalty * sls_tgt_dev_abs

    # Constraints
    model += lpSum(adjusted_sales) + sls_tgt_deviation == adjusted_sales_target
    model += sls_tgt_dev_abs >= sls_tgt_deviation
    model += sls_tgt_dev_abs >= -sls_tgt_deviation

    for i in range(start_month, required_months):
        idx = i - start_month

        if frozen_months[i] == 1:
            model += (
                adjusted_sales[idx] == original_sales[i]
            )  # If sales are frozen, force adjusted sales to equal original sales
        else:
            model += adjusted_sales[idx] >= R_minus * original_sales[i]
            model += adjusted_sales[idx] <= R_plus * original_sales[i]

        # Inventory flow
        if i == start_month:
            model += (
                inventory[idx]
                == initial_inventory + wholesales[i] - adjusted_sales[idx]
            )
        else:
            model += (
                inventory[idx]
                == inventory[idx - 1] + wholesales[i] - adjusted_sales[idx]
            )

        # Min and max days of supply constraints
        model += (
            inventory[idx] * selling_days[i] >= min_dos_targets[i] * adjusted_sales[idx]
        )
        model += (
            inventory[idx] * selling_days[i] <= max_dos_targets[i] * adjusted_sales[idx]
        )

    # Solve
    solver = PULP_CBC_CMD(msg=True, gapRel=0.001, timeLimit=10)
    model.solve(solver)


    # Retrieve results
    final_sales = [
        (
            original_sales[i]
            if frozen_months[i] == 1
            else adjusted_sales[i - start_month].varValue
        )
        for i in range(required_months)
    ]
    final_inventory = [
        dealer_stock[i] if i < start_month else inventory[i - start_month].varValue
        for i in range(required_months)
    ]
    monthly_ratios = [
        final_sales[i] / original_sales[i] if original_sales[i] != 0 else None
        for i in range(required_months)
    ]
    monthly_dos = [
        (
            final_inventory[i] / (final_sales[i] / selling_days[i])
            if final_sales[i] > 0
            else None
        )
        for i in range(required_months)
    ]

    return {
        "final_sales": final_sales,
        "R_plus": R_plus.varValue,
        "R_minus": R_minus.varValue,
        "ratios": monthly_ratios,
        "target_deviation": sls_tgt_deviation.varValue,
        "inventory_levels": final_inventory,
        "days_of_supply": monthly_dos,
    }