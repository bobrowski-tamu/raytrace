# Super Simple Git Setup Guide

1. Install Git
2. Download the project to your computer
3. Make a small change
4. Save and upload that change

## 1) Install Git

### Windows
- Go to the Git website
- Download **Git for Windows** 
- Install it with the default options

### Mac
- Open **Terminal**
- Type:

```bash
git --version
```

- If Git is not installed, your Mac will usually prompt you to install it
- Hint: Use homebrew to install git on mac (if you use homebrew, which i recommend)


## 2) Open Terminal / Command Line

You will need a terminal.

- **Windows:** open **Git Bash**
- **Mac:** open **Terminal**

## 3) Tell Git your name and email (ideally Github email)

Run these commands once, replacing with your real info:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Example:

```bash
git config --global user.name "Jane Smith"
git config --global user.email "jane@company.com"
```

## 4) Clone the repository

I will send you the repo URL. It will look like this:

```bash
https://github.com/bobrowski-tamu/raytrace.git
```

In the terminal, go to the place where you want the project folder, then run:

```bash
git clone https://github.com/bobrowski-tamu/raytrace.git
```

This downloads the repo to your computer.

Then move into the folder:

```bash
cd REPO
```

Example:

```bash
cd myproject
```

## 5) Make a first change

Open the project folder on your computer.

Make changes.

Save the file.

## 6) Check what changed

In terminal, inside the repo folder, run:

```bash
git status
```

You should see the changed file.

## 7) Stage the change

Run:

```bash
git add .
```

This prepares your change to be saved.

## 8) Make your first commit

Run:

```bash
git commit -m "My first commit"
```

This saves your change locally.

## 9) Upload the change to GitHub

Run:

```bash
git push
```

You may be asked to sign in to GitHub in the browser.

# Full example

```bash
git config --global user.name "Jane Smith"
git config --global user.email "jane@company.com"
git clone https://github.com/OWNER/REPO.git
cd REPO
git status
git add .
git commit -m "My first commit"
git push
```

# Everyday workflow after that

Each time you work:

```bash
git pull
```

Make your changes, then:

```bash
git add .
git commit -m "describe what you changed"
git push
```

# If something goes wrong

## “git: command not found”
Git is not installed yet.

## “Permission denied” or login problems
Make sure:
- you accepted the GitHub repo invite
- you are signed into the correct GitHub account

## “nothing to commit”
You did not change any files yet.
