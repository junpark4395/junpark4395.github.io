import { getCollection, type CollectionEntry } from 'astro:content';

import { firstParagraph } from './excerpt';
import { slugify } from './slugify';

export type Post = CollectionEntry<'posts'>;

/** Published posts, newest first. Drafts show up in `bun run dev` only. */
export async function getPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', (p) => import.meta.env.DEV || !p.data.draft);
  return posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

/** URL slug: content path minus extension. 바꾸면 글 주소와 Disqus 스레드가 끊긴다. */
export function postSlug(post: Post): string {
  return post.id.replace(/\.md$/i, '');
}

export const postPath = (post: Post) => `/posts/${postSlug(post)}/`;

export const tagPath = (tag: string) => `/tags/${slugify(tag)}/`;

/** Unique tags (first spelling wins per slug), alphabetical. */
export function allTags(posts: Post[]): string[] {
  const bySlug = new Map<string, string>();
  for (const tag of posts.flatMap((p) => p.data.tags)) {
    if (!bySlug.has(slugify(tag))) bySlug.set(slugify(tag), tag);
  }
  return [...bySlug.values()].sort((a, b) => a.localeCompare(b));
}

export const excerpt = (post: Post) => post.data.description ?? firstParagraph(post.body ?? '');

/** "Oct 5, 2026". UTC so a date-only pubDate never shifts a day with the build machine's zone. */
export const formatDate = (date: Date) =>
  date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  });
