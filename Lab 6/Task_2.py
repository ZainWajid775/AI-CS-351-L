import matplotlib.pyplot as plt

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution = None

    def solve(self):
        return self.backtrack({})

    def backtrack(self, assignment):
        if len(assignment) == len(self.variables):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var):
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                del assignment[var]

        return None

    def select_unassigned_variable(self, assignment):
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: len(self.domains[v]))

    def order_domain_values(self, var):
        return self.domains[var]

    def is_consistent(self, var, value, assignment):
        time, room = value
        for other_var in assignment:
            other_time, other_room = assignment[other_var]

            if self.constraints["instructor"][var] == self.constraints["instructor"][other_var]:
                if time == other_time:
                    return False

            if set(self.constraints["students"][var]) & set(self.constraints["students"][other_var]):
                if time == other_time:
                    return False

            if room == other_room and time == other_time:
                return False

        if self.constraints["room_capacity"][room] < self.constraints["enrollment"][var]:
            return False

        return True


# --- Example usage ---
if __name__ == "__main__":
    variables = ["CS101", "MATH201", "PHY301"]

    domains = {
        "CS101": [("Mon9am", "R1"), ("Tue11am", "R2")],
        "MATH201": [("Mon9am", "R2"), ("Tue11am", "R1")],
        "PHY301": [("Mon9am", "R3"), ("Tue11am", "R2")]
    }

    constraints = {
        "instructor": {"CS101": "Dr. A", "MATH201": "Dr. B", "PHY301": "Dr. A"},
        "students": {
            "CS101": {"Ali", "Sara", "Bilal"},
            "MATH201": {"Sara", "Usman"},
            "PHY301": {"Ali", "Bilal"}
        },
        "room_capacity": {"R1": 40, "R2": 25, "R3": 30},
        "enrollment": {"CS101": 30, "MATH201": 20, "PHY301": 25}
    }

    scheduler = CSP(variables, domains, constraints)
    solution = scheduler.solve()

    if solution:
        print("Schedule found:")
        for course, (time, room) in solution.items():
            print(f"{course}: {time} in {room}")
    else:
        print("No valid schedule found.")

    # --- Visualization ---
    if solution:
        times = sorted(set([v[0] for v in sum(domains.values(), [])]))
        rooms = sorted(set([v[1] for v in sum(domains.values(), [])]))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("University Course Schedule", fontsize=14, fontweight="bold")

        # Create grid
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels(times)
        ax.set_yticks(range(len(rooms)))
        ax.set_yticklabels(rooms)
        ax.set_xlim(-0.5, len(times) - 0.5)
        ax.set_ylim(-0.5, len(rooms) - 0.5)
        ax.grid(True)

        # Plot each scheduled course
        for course, (time, room) in solution.items():
            x = times.index(time)
            y = rooms.index(room)
            ax.text(x, y, f"{course}\n{constraints['instructor'][course]}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='skyblue', edgecolor='black', boxstyle='round,pad=0.3'))

        plt.tight_layout()
        plt.show()