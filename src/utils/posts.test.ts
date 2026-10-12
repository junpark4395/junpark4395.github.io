import { describe, expect, test } from 'bun:test';
import { firstParagraph } from './excerpt';
import { slugify } from './slugify';

describe('slugify', () => {
  test('Latin tags remain unchanged', () => {
    expect(slugify('JavaScript')).toBe('javascript');
  });

  test('Japanese tags work', () => {
    expect(slugify('テスト')).toBe('テスト');
  });

  test('Malayalam tags work', () => {
    expect(slugify('മലയാളം')).toBe('മലയാളം');
  });

  test('Mixed strings work', () => {
    expect(slugify('日本語 テスト')).toBe('日本語-テスト');
  });

  test('Multiple spaces become single hyphen', () => {
    expect(slugify('hello   world')).toBe('hello-world');
  });

  test('Special characters are removed', () => {
    expect(slugify('hello@world!')).toBe('helloworld');
  });

  test('Leading and trailing hyphens are trimmed', () => {
    expect(slugify('  hello  ')).toBe('hello');
  });

  test('Mixed Latin and Unicode', () => {
    expect(slugify('React テスト')).toBe('react-テスト');
  });

  test('Accented characters are preserved', () => {
    expect(slugify('café')).toBe('café');
  });

  test('Empty string after processing returns empty', () => {
    expect(slugify('!!!')).toBe('');
  });
});

describe('firstParagraph', () => {
  test('skips headings and strips link syntax', () => {
    const md = "## Intro\n\nI'm picking up AI again: [karpathy/micrograd](https://x.y)\n\nSecond.";
    expect(firstParagraph(md)).toBe("I'm picking up AI again: karpathy/micrograd");
  });

  test('joins soft-wrapped lines and drops bold/code marks', () => {
    expect(firstParagraph('Trained **one neuron**\nwith `Value`.\n\nMore.')).toBe(
      'Trained one neuron with Value.',
    );
  });

  test('skips images, code fences and display math', () => {
    const md =
      '![g](g.png)\n\n```cpp\nint x;\n\nint y;\n```\n\n$$\ne = ab\n$$\n\n$$e=ab$$\n\nReal text.';
    expect(firstParagraph(md)).toBe('Real text.');
  });

  test('keeps underscores inside identifiers', () => {
    expect(firstParagraph('See micrograd_cpp and snake_case.')).toBe(
      'See micrograd_cpp and snake_case.',
    );
  });

  test('empty body gives empty string', () => {
    expect(firstParagraph('## Only a heading')).toBe('');
  });
});
