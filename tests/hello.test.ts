import { expect, describe, it } from 'bun:test'
import { writeLog } from '../scripts'

describe('#EXEMPLE_TEST', () => {
  it('Write a Hello log', async () => {
    const promise = new Promise<boolean>((resolve) => resolve(true))

    const res = await promise

    writeLog({
      content: 'Hello',
      label: 'EXEMPLE_TEST',
      format: 'json',
    })

    expect(res).toBeTruthy()
  })
})
