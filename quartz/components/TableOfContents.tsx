import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import legacyStyle from "./styles/legacyToc.scss"
import modernStyle from "./styles/toc.scss"
import { classNames } from "../util/lang"

// @ts-ignore
import script from "./scripts/toc.inline"
import { i18n } from "../i18n"
import OverflowListFactory from "./OverflowList"
import { concatenateResources } from "../util/resources"

interface Options {
  layout: "modern" | "legacy"
}

const defaultOptions: Options = {
  layout: "modern",
}

interface TocEntry {
  depth: number
  text: string
  slug: string
}

interface TocNode {
  entry: TocEntry
  children: TocNode[]
}

function buildTocTree(toc: TocEntry[]): TocNode[] {
  const roots: TocNode[] = []
  const stack: TocNode[] = []
  for (const entry of toc) {
    const node: TocNode = { entry, children: [] }
    while (stack.length && stack[stack.length - 1].entry.depth >= entry.depth) {
      stack.pop()
    }
    if (stack.length === 0) {
      roots.push(node)
    } else {
      stack[stack.length - 1].children.push(node)
    }
    stack.push(node)
  }
  return roots
}

const Chevron = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    class="fold"
  >
    <polyline points="6 9 12 15 18 9"></polyline>
  </svg>
)

const renderNode = (node: TocNode, isRoot: boolean) => {
  const hasChildren = node.children.length > 0
  const startCollapsed = hasChildren && !isRoot
  const liClass = `depth-${node.entry.depth}${startCollapsed ? " collapsed" : ""}`
  return (
    <li key={node.entry.slug} class={liClass}>
      <div class="toc-entry">
        {hasChildren ? (
          <button
            type="button"
            class="toc-entry-toggle"
            aria-expanded={startCollapsed ? "false" : "true"}
          >
            <Chevron />
          </button>
        ) : (
          <span class="toc-entry-spacer" />
        )}
        <a href={`#${node.entry.slug}`} data-for={node.entry.slug}>
          {node.entry.text}
        </a>
      </div>
      {hasChildren && (
        <ul class="toc-children">{node.children.map((c) => renderNode(c, false))}</ul>
      )}
    </li>
  )
}

let numTocs = 0
export default ((opts?: Partial<Options>) => {
  const layout = opts?.layout ?? defaultOptions.layout
  const { OverflowList, overflowListAfterDOMLoaded } = OverflowListFactory()
  const TableOfContents: QuartzComponent = ({
    fileData,
    displayClass,
    cfg,
  }: QuartzComponentProps) => {
    if (!fileData.toc) {
      return null
    }

    const id = `toc-${numTocs++}`
    return (
      <div class={classNames(displayClass, "toc")}>
        <button
          type="button"
          class={fileData.collapseToc ? "collapsed toc-header" : "toc-header"}
          aria-controls={id}
          aria-expanded={!fileData.collapseToc}
        >
          <h3>{i18n(cfg.locale).components.tableOfContents.title}</h3>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="fold"
          >
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </button>
        <OverflowList
          id={id}
          class={fileData.collapseToc ? "collapsed toc-content" : "toc-content"}
        >
          {buildTocTree(fileData.toc).map((node) => renderNode(node, true))}
        </OverflowList>
      </div>
    )
  }

  TableOfContents.css = modernStyle
  TableOfContents.afterDOMLoaded = concatenateResources(script, overflowListAfterDOMLoaded)

  const LegacyTableOfContents: QuartzComponent = ({ fileData, cfg }: QuartzComponentProps) => {
    if (!fileData.toc) {
      return null
    }
    return (
      <details class="toc" open={!fileData.collapseToc}>
        <summary>
          <h3>{i18n(cfg.locale).components.tableOfContents.title}</h3>
        </summary>
        <ul>
          {fileData.toc.map((tocEntry) => (
            <li key={tocEntry.slug} class={`depth-${tocEntry.depth}`}>
              <a href={`#${tocEntry.slug}`} data-for={tocEntry.slug}>
                {tocEntry.text}
              </a>
            </li>
          ))}
        </ul>
      </details>
    )
  }
  LegacyTableOfContents.css = legacyStyle

  return layout === "modern" ? TableOfContents : LegacyTableOfContents
}) satisfies QuartzComponentConstructor
