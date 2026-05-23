// Focused popover for IEEE citation links (`a[href^="#ref-"]`).
//
// Quartz's default popover fetches the whole page and scrolls — for a
// citation that means the reader sees a chunk of the Sources section
// rather than the specific entry. This script:
//   1. marks citation links with data-no-popover so Quartz's default skips
//   2. groups adjacent citation links (e.g. "[12], [17]") into clusters
//   3. attaches a hover handler per cluster that shows ONLY the referenced
//      Sources entries
//   4. keeps the popover open when the mouse moves onto it so the reader
//      can follow URLs inside

let popoverEl: HTMLDivElement | null = null
let hideTimer: number | null = null
const HIDE_DELAY_MS = 200

function ensurePopover(): HTMLDivElement {
  if (popoverEl && document.body.contains(popoverEl)) return popoverEl
  popoverEl = document.createElement("div")
  popoverEl.className = "cite-popover"
  popoverEl.addEventListener("mouseenter", cancelHide)
  popoverEl.addEventListener("mouseleave", scheduleHide)
  document.body.appendChild(popoverEl)
  return popoverEl
}

function cancelHide() {
  if (hideTimer !== null) {
    window.clearTimeout(hideTimer)
    hideTimer = null
  }
}

function scheduleHide() {
  cancelHide()
  hideTimer = window.setTimeout(() => {
    if (popoverEl) popoverEl.classList.remove("active")
    hideTimer = null
  }, HIDE_DELAY_MS)
}

function isCitationLink(el: Element): el is HTMLAnchorElement {
  return (
    el.tagName === "A" &&
    (el as HTMLAnchorElement).getAttribute("href")?.startsWith("#ref-") === true
  )
}

function groupClusters(links: HTMLAnchorElement[]): HTMLAnchorElement[][] {
  const clusters: HTMLAnchorElement[][] = []
  const seen = new WeakSet<HTMLAnchorElement>()
  for (const link of links) {
    if (seen.has(link)) continue
    const cluster: HTMLAnchorElement[] = [link]
    seen.add(link)
    let next: Node | null = link.nextSibling
    while (next) {
      if (next.nodeType === Node.TEXT_NODE) {
        const text = next.textContent || ""
        if (!/^[\s,]+$/.test(text)) break
        next = next.nextSibling
        continue
      }
      if (next.nodeType === Node.ELEMENT_NODE && isCitationLink(next as Element)) {
        const adj = next as HTMLAnchorElement
        cluster.push(adj)
        seen.add(adj)
        next = adj.nextSibling
        continue
      }
      break
    }
    clusters.push(cluster)
  }
  return clusters
}

function showFor(cluster: HTMLAnchorElement[], anchor: HTMLAnchorElement) {
  const el = ensurePopover()
  cancelHide()
  el.innerHTML = ""
  let any = false
  for (const link of cluster) {
    const href = link.getAttribute("href")
    if (!href || !href.startsWith("#")) continue
    const id = href.slice(1)
    const target = document.getElementById(id)
    if (!target) continue
    const clone = target.cloneNode(true) as HTMLElement
    clone.removeAttribute("id")
    el.appendChild(clone)
    any = true
  }
  if (!any) return

  el.classList.add("active")
  // Position: default below the link; flip above if it would overflow.
  const rect = anchor.getBoundingClientRect()
  const margin = 8
  const popoverHeight = el.offsetHeight
  const popoverWidth = el.offsetWidth
  let top = rect.bottom + margin
  if (top + popoverHeight > window.innerHeight) {
    top = Math.max(margin, rect.top - popoverHeight - margin)
  }
  let left = rect.left
  if (left + popoverWidth > window.innerWidth - margin) {
    left = Math.max(margin, window.innerWidth - popoverWidth - margin)
  }
  el.style.top = `${top}px`
  el.style.left = `${left}px`
}

document.addEventListener("nav", () => {
  const links = Array.from(
    document.querySelectorAll<HTMLAnchorElement>('a[href^="#ref-"]'),
  )
  for (const link of links) link.dataset.noPopover = "true"

  const clusters = groupClusters(links)
  for (const cluster of clusters) {
    const onEnter = (ev: MouseEvent) => showFor(cluster, ev.currentTarget as HTMLAnchorElement)
    for (const link of cluster) {
      link.addEventListener("mouseenter", onEnter)
      link.addEventListener("mouseleave", scheduleHide)
      window.addCleanup(() => {
        link.removeEventListener("mouseenter", onEnter)
        link.removeEventListener("mouseleave", scheduleHide)
      })
    }
  }
})
