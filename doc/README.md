# ec-MLP Documentation Deployment

This directory contains the documentation for ec-MLP, built using [mdBook](https://rust-lang.github.io/mdBook/).

## Local Development

To build the documentation locally:

1. Install mdBook:
   ```bash
   # Download and install mdBook (Linux x86_64)
   curl -L https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-x86_64-unknown-linux-gnu.tar.gz | tar xz
   sudo mv mdbook /usr/local/bin/
   ```

2. Build the documentation:
   ```bash
   cd doc
   mdbook build
   ```

3. Serve locally for development:
   ```bash
   cd doc
   mdbook serve
   ```
   The documentation will be available at `http://localhost:3000`

## GitHub Pages Deployment

The documentation is automatically deployed to GitHub Pages using GitHub Actions workflows:

### Workflow Files

1. **`.github/workflows/deploy-docs-ghpages.yml`** - Main deployment workflow
   - Triggers on pushes to `main` and `devel` branches
   - Builds documentation using mdBook
   - Deploys to the `gh-pages` branch

2. **`.github/workflows/deploy-docs.yml`** - Alternative deployment workflow
   - Uses GitHub Pages deployment action
   - More advanced configuration with permissions

### Setup Instructions

1. **Enable GitHub Pages** in your repository:
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` and `/ (root)`

2. **Configure GitHub Actions permissions**:
   - Go to repository Settings → Actions → General
   - Workflow permissions: Read and write permissions
   - Allow GitHub Actions to create and approve pull requests

3. **The workflows will automatically**:
   - Build the documentation when changes are pushed to `doc/` directory
   - Deploy to GitHub Pages
   - Update the live site

### Manual Deployment

To manually trigger a deployment:

1. Go to the Actions tab in your GitHub repository
2. Select "Deploy Documentation to GitHub Pages" workflow
3. Click "Run workflow"

### Documentation Structure

```
doc/
├── src/                    # Source markdown files
│   ├── SUMMARY.md         # Table of contents
│   ├── data_modifier.md   # Data modifier documentation
│   ├── lmp/               # LAMMPS interface docs
│   └── tf/                # TensorFlow modifier docs
├── book.toml              # mdBook configuration
├── ec-mlp-doc/            # Generated HTML files (auto-generated)
└── README.md              # This file
```

### Customization

To customize the documentation:

1. Edit `book.toml` for book configuration
2. Add new markdown files to `src/`
3. Update `SUMMARY.md` to include new pages in the navigation
4. The documentation will be automatically rebuilt and deployed

### Troubleshooting

If the documentation doesn't deploy:

1. Check the Actions tab for workflow errors
2. Ensure GitHub Pages is enabled in repository settings
3. Verify the workflow has proper permissions
4. Check that the documentation builds locally with `mdbook build`

For more information about mdBook, see the [official documentation](https://rust-lang.github.io/mdBook/).