import { QuartzComponentConstructor } from "./types"
import script from "./scripts/cite-popover.inline"
// @ts-ignore — scss imports are processed by Quartz's build
import style from "./styles/cite-popover.scss"

function CitePopover() {
  return null
}

CitePopover.css = style
CitePopover.afterDOMLoaded = script

export default (() => CitePopover) satisfies QuartzComponentConstructor
