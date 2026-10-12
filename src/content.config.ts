import { glob } from 'astro/loaders';
import { defineCollection } from 'astro:content';
import { z } from 'zod';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string().min(1).max(140),
    /** 비우면 본문 첫 문단이 목록 요약으로 쓰인다. */
    description: z.string().max(280).optional(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    math: z.boolean().default(false),
    comments: z.boolean().optional(),
    /** 예전 사이트에서 옮겨 온 글의 Disqus 스레드 식별자. */
    disqusId: z.string().optional(),
  }),
});

const pages = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/pages' }),
  schema: z.object({
    title: z.string().min(1),
    description: z.string().optional(),
  }),
});

export const collections = { posts, pages };
