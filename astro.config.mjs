// @ts-check
import { satteri, satteriHeadingIdsPlugin } from '@astrojs/markdown-satteri';
import sitemap from '@astrojs/sitemap';
import expressiveCode from 'astro-expressive-code';
import { defineConfig } from 'astro/config';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import satteriExternalLinks from 'satteri-external-links';
import { katex } from '@nullpinter/satteri-katex';

import { SITE } from './src/config';

const SITEMAP_XSL_HREF = '/sitemap/styles.xsl';

/**
 * Tiny inline integration: after `@astrojs/sitemap` runs, rewrite the
 * absolute XSL `href` it emits to a root-relative path so the styled
 * sitemap also renders under `bun preview` (no cross-origin XSLT block).
 * Crawlers ignore `<?xml-stylesheet ?>` entirely, so SEO is unaffected.
 */
function rewriteSitemapXslToRelative() {
  return {
    name: 'rewrite-sitemap-xsl',
    hooks: {
      'astro:build:done': (/** @type {{ dir: URL }} */ { dir }) => {
        const distDir = fileURLToPath(dir);
        const files = readdirSync(distDir).filter(
          (f) => f.startsWith('sitemap') && f.endsWith('.xml'),
        );
        for (const file of files) {
          const path = join(distDir, file);
          const xml = readFileSync(path, 'utf8');
          const fixed = xml.replace(
            /<\?xml-stylesheet\b[^?]*\?>/,
            `<?xml-stylesheet type="text/xsl" href="${SITEMAP_XSL_HREF}"?>`,
          );
          if (fixed !== xml) writeFileSync(path, fixed);
        }
      },
    },
  };
}

// https://astro.build/config
export default defineConfig({
  site: SITE.url,
  // 예전 URL 체계. 색인·외부 링크가 끊기지 않게 유지한다.
  redirects: {
    '/posts/Life-2026-02-01': '/posts/2026-02-01-a-new-car/',
    '/journal/2026-02-01-a-new-car': '/posts/2026-02-01-a-new-car/',
    '/writing/2026-02-01-a-new-car': '/posts/2026-02-01-a-new-car/',
    '/writing': '/',
  },
  trailingSlash: 'ignore',
  build: {
    format: 'directory',
  },
  prefetch: true,

  image: {
    layout: 'constrained',
    responsiveStyles: true,
  },

  markdown: {
    // KaTeX renders math at build time; its stylesheet is loaded only on
    // `math: true` posts via `<MathStyles />`.
    processor: satteri({
      features: { math: true },
      mdastPlugins: [katex()],
      hastPlugins: [
        satteriHeadingIdsPlugin(),
        satteriExternalLinks({
          target: '_blank',
          rel: ['nofollow', 'noopener', 'noreferrer'],
        }),
      ],
    }),
  },

  integrations: [
    // Syntax highlighting (Shiki) with frames, titles and a copy button.
    // Light/dark theme follows `prefers-color-scheme`.
    // https://expressive-code.com/
    expressiveCode({
      themes: ['github-light', 'github-dark-dimmed'],
      shiki: {
        langAlias: {
          env: 'dotenv',
        },
      },
      styleOverrides: {
        borderRadius: '3px',
        codeFontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
        codeFontSize: '0.875rem',
        frames: {
          shadowColor: 'transparent',
        },
      },
    }),
    sitemap({
      // Browsers apply this XSL to render a readable sitemap; crawlers ignore it.
      xslURL: SITEMAP_XSL_HREF,
      filter: (page) => !page.endsWith('/404/'),
    }),
    rewriteSitemapXslToRelative(),
  ],

  experimental: {
    contentIntellisense: true,
  },
});
