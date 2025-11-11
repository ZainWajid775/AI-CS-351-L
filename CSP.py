# Simple CSP: Map coloring using backtracking

# Each region must be colored differently
neighbors = {
    "WA": ["NT", "SA"],
    "NT": ["WA", "SA", "Q"],
    "SA": ["WA", "NT", "Q", "NSW", "V"],
    "Q": ["NT", "SA", "NSW"],
    "NSW": ["Q", "SA", "V"],
    "V": ["SA", "NSW"],
    "T": [],  # Isolated (no neighbors)
}

colors = ["Red", "Green", "Blue"]  # Available Colors
assignment = {}  # Holds the color chosen for each region


def isValid(region, color):
    for neighbor in neighbors[region]:
        if neighbor in assignment and assignment[neighbor] == color:
            return False

    return True


def backtrack():
    if len(assignment) == len(neighbors):  # ALl regions colored
        return True

    region = [r for r in neighbors if r not in assignment][0]  # Pick unassignd region

    for color in colors:
        if isValid(region, color):
            assignment[region] = color
            if backtrack():  # Recursion
                return True
            del assignment[region]
    return False  # No color found


# Run the CSP solver
backtrack()
print("Solution", assignment)
