import pandas as pd

data = pd.DataFrame({"text": ["Hello world!", "123 Main St", "This is some text."]})

# Check for object dtype
if pd.api.types.is_object_dtype(data["text"]):
  print("The 'text' column might contain textual data.")

# Check for alphanumeric characters
is_alnum = data["text"].str.isalnum()
print(f"Checking : {is_alnum}")
print("'text' column has alphanumeric characters:", is_alnum.all())

# Use regular expression (example: check for digits at the beginning)
import re
pattern = r"^\d+"
has_digits_at_start = data["text"].str.contains(pattern)
print("'text' column has digits at the start:", has_digits_at_start.any())
