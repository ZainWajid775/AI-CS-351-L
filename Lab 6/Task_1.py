import matplotlib.pyplot as plt
import math

# -----------------------------
# Problem setup
# -----------------------------

guests = ["A", "B", "C", "D", "E", "F"]
tables = {"T1": 3, "T2": 3}  # table name: capacity

# Constraints
prefer_together = [("A", "B")]       # A wants to sit with B
avoid = [("C", "D")]                 # C avoids D
special_needs = {"E": ["T2"]}        # E must be at T2

# -----------------------------
# CSP Backtracking Solver
# -----------------------------

def is_valid(assignment, guest, table):
    # Check table capacity
    table_counts = {t: list(assignment.values()).count(t) for t in tables}
    if table_counts.get(table, 0) >= tables[table]:
        return False

    # Avoidance constraint
    for (g1, g2) in avoid:
        if guest == g1 and g2 in assignment and assignment[g2] == table:
            return False
        if guest == g2 and g1 in assignment and assignment[g1] == table:
            return False

    # Special needs
    if guest in special_needs and table not in special_needs[guest]:
        return False

    return True


def backtrack(assignment):
    if len(assignment) == len(guests):
        return assignment

    guest = [g for g in guests if g not in assignment][0]

    for table in tables:
        if is_valid(assignment, guest, table):
            assignment[guest] = table
            result = backtrack(assignment)
            if result:
                return result
            del assignment[guest]
    return None


assignment = backtrack({})

# -----------------------------
# Scoring satisfied preferences
# -----------------------------
score = 0
for (g1, g2) in prefer_together:
    if assignment.get(g1) == assignment.get(g2):
        score += 1

print("Optimal Seating Arrangement:")
for g, t in assignment.items():
    print(f"Guest {g} → {t}")
print(f"Preference satisfaction score: {score}")

# -----------------------------
# Matplotlib Visualization
# -----------------------------

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Event Seating Arrangement", fontsize=14)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Table coordinates for layout
table_positions = {"T1": (3, 2.5), "T2": (7, 2.5)}

# Draw tables
for table, (x, y) in table_positions.items():
    circle = plt.Circle((x, y), 1.0, color="lightblue", ec="blue", lw=2)
    ax.add_patch(circle)
    ax.text(x, y, table, ha="center", va="center", fontsize=12, weight="bold")

# Assign guest positions around each table
for table, (x, y) in table_positions.items():
    guests_at_table = [g for g, t in assignment.items() if t == table]
    n = len(guests_at_table)
    if n == 0:
        continue

    # Distribute guests evenly around the circle
    for i, g in enumerate(guests_at_table):
        angle = 2 * math.pi * i / n
        gx = x + 1.5 * math.cos(angle)
        gy = y + 1.5 * math.sin(angle)
        ax.text(gx, gy, g, ha="center", va="center",
                fontsize=11, bbox=dict(facecolor="pink", edgecolor="black", boxstyle="circle"))

plt.show()
