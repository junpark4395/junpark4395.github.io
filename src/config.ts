import process from 'node:process';

export const SITE = {
  title: 'Jun Park',
  description: '',
  /** Public URL of the deployed site, no trailing slash. Breaks canonical/RSS/sitemap if wrong. */
  // `||` (not `??`) so an explicitly empty `SITE_URL=` also falls back.
  url: process.env.SITE_URL || 'https://junpark4395.github.io',
  github: 'junpark4395',
  postsPerPage: 8,
};

/** Disqus 댓글. 예전 사이트의 스레드를 그대로 잇는다. */
export const DISQUS = {
  enabled: true,
  shortname: 'junpark',
} as const;
