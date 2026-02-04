
def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    count = len(numbers)
    return total / count

def main():
    sample_data = [10, 20, 30, 40, 50]
    avg = calculate_average(sample_data)
    print(f"Average: {avg}")

if __name__ == "__main__":
    main()