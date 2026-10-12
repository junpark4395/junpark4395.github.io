import rss from '@astrojs/rss';
import type { APIRoute } from 'astro';
import { SITE } from '../config';
import { excerpt, getPosts, postPath } from '../utils/posts';

export const GET: APIRoute = async () =>
  rss({
    title: SITE.title,
    description: SITE.description,
    site: SITE.url,
    stylesheet: '/rss/styles.xsl',
    items: (await getPosts()).map((post) => ({
      title: post.data.title,
      pubDate: post.data.pubDate,
      description: excerpt(post),
      link: postPath(post),
      categories: post.data.tags,
    })),
    customData: '<language>en-us</language>',
  });
