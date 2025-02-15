import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { ProblemData } from '../../types';

const dataDir = path.join(process.cwd(), 'data');

export const GET = async (request: Request) => {
  const { searchParams } = new URL(request.url);
  const files = searchParams.getAll('files');

  if (files.length > 0) {
    let combinedData: ProblemData[] = [];
    for (const file of files) {
      const filePath = path.join(dataDir, `${file}.json`);
      const data: ProblemData[] = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      combinedData = combinedData.concat(data);
    }
    return NextResponse.json(combinedData);
  } else {
    const fileList = fs.readdirSync(dataDir).filter((file) => file.endsWith('.json')).map((file) => path.parse(file).name);
    return NextResponse.json(fileList);
  }
}