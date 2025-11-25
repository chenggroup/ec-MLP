# ec-MLP Documentation Deployment Guide

This guide provides step-by-step instructions for deploying the ec-MLP documentation to GitHub Pages.

## Prerequisites

- GitHub repository with ec-MLP code
- Admin access to the repository
- GitHub account with appropriate permissions

## Quick Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Build and deployment", set:
   - Source: **Deploy from a branch**
   - Branch: **gh-pages**
   - Folder: **/(root)**
4. Click **Save**

### 2. Configure Actions Permissions

1. Go to **Settings** → **Actions** → **General**
2. Scroll to "Workflow permissions"
3. Select **Read and write permissions**
4. Check **Allow GitHub Actions to create and approve pull requests**
5. Click **Save**

### 3. Push Changes

The workflows are already configured in the repository. Simply push your changes to trigger the deployment:

```bash
git add .
git commit -m "Add documentation deployment setup"
git push origin main
```

## What Happens Next

1. **Automatic Build**: When you push to `main` or `devel` branches, the documentation will automatically build
2. **Deployment**: The built documentation will be deployed to the `gh-pages` branch
3. **Live Site**: Your documentation will be available at: `https://[your-username].github.io/ec-MLP/`

## Manual Deployment

If you need to trigger a deployment manually:

1. Go to the **Actions** tab in your repository
2. Select **"Deploy Documentation to GitHub Pages"** workflow
3. Click **"Run workflow"**
4. Choose the branch and click **"Run workflow"**

## Local Development

To work on documentation locally:

```bash
# Install mdBook
curl -L https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-x86_64-unknown-linux-gnu.tar.gz | tar xz
sudo mv mdbook /usr/local/bin/

# Build and serve locally
cd doc
mdbook serve
```

Visit `http://localhost:3000` to see your changes.

## File Structure

```
ec-MLP/
├── .github/workflows/
│   ├── deploy-docs.yml          # GitHub Pages deployment (advanced)
│   ├── deploy-docs-ghpages.yml  # gh-pages deployment (recommended)
│   └── ci.yml                   # CI with documentation build
├── doc/
│   ├── src/                     # Documentation source files
│   ├── book.toml               # mdBook configuration
│   ├── ec-mlp-doc/             # Generated HTML (auto-generated)
│   └── README.md               # Documentation development guide
├── DEPLOYMENT_GUIDE.md         # This file
└── README.md                   # Main project README
```

## Troubleshooting

### Documentation Not Appearing

1. Check the **Actions** tab for workflow errors
2. Ensure GitHub Pages is enabled in Settings
3. Verify the `gh-pages` branch exists
4. Check that the workflow has proper permissions

### Build Errors

1. Run `mdbook build` locally to test
2. Check for syntax errors in markdown files
3. Verify `SUMMARY.md` is properly formatted
4. Ensure all referenced files exist

### Permission Issues

1. Go to Settings → Actions → General
2. Enable "Read and write permissions"
3. Check repository access for Actions

## Customization

### Changing Documentation URL

To use a custom domain:

1. Create a `CNAME` file in `doc/src/`
2. Add your domain to the file
3. Update GitHub Pages settings to use custom domain

### Styling Changes

1. Edit `book.toml` for basic configuration
2. Add custom CSS in `doc/src/`
3. Modify the theme as needed

## Support

For issues with the documentation deployment:

1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review the [mdBook guide](https://rust-lang.github.io/mdBook/)
3. Open an issue in the ec-MLP repository

## Workflow Details

The deployment process uses two main workflows:

1. **deploy-docs-ghpages.yml** (Recommended)
   - Simple and reliable
   - Uses peaceiris/actions-gh-pages action
   - Deploys to gh-pages branch

2. **deploy-docs.yml** (Advanced)
   - Uses GitHub Pages deployment API
   - More configuration options
   - Requires additional permissions

Both workflows will:

- Install mdBook
- Build the documentation
- Deploy to GitHub Pages
- Update automatically on changes to `doc/` directory
