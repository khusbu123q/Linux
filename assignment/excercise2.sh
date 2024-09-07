#!/bin/bash

while read -r line; do
  #extracting userid, username, and department from the line
  userid=$(echo "$line" | cut -d',' -f1)
  username_dept=$(echo "$line" | cut -d',' -f2)
  username=$(echo "$username_dept" | cut -d':' -f1)
  dept=$(echo "$username_dept" | cut -d':' -f2)

  #displaying the extracted information
  echo "User ID: $userid"
  echo "Username: $username"
  echo "Department: $dept"

done < users.csv

