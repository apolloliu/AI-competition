#!/bin/dash

git init
echo "# AI-competition" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:apolloliu/AI-competition.git
git push -u origin main
