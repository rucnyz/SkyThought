"use client";
import React, { useState, useEffect, useCallback } from 'react';
import { MathJax, MathJaxContext } from 'better-react-mathjax';
import { ProblemData } from './types';
import { Select, SelectItem } from '@heroui/select';
import { Selection } from '@heroui/react';

const mathJaxConfig = { loader: { load: ['input/tex', 'output/chtml'] }, tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] } };
const ITEMS_PER_PAGE = 2;


const FileSelector = React.memo(({ onSelect }: { onSelect: (files: string[]) => void }) => {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedFiles, setSelectedFiles] = React.useState<Selection>(new Set());

  useEffect(() => {
    fetch('/api/data').then(res => res.json()).then(setDatasets);
  }, []);

  const handleChange = (keys: Selection) => {
    setSelectedFiles(keys);
    onSelect(Array.from(keys).map(key => key.toString()));
  };

  return (
    <div className="flex w-full flex-col gap-2">
      <Select
        className="w-full"
        label="Select Datasets"
        placeholder="Select datasets"
        selectedKeys={selectedFiles}
        variant="bordered"
        selectionMode="multiple"
        onSelectionChange={handleChange}
      >
        {datasets.map((dataset) => (
          <SelectItem key={dataset} textValue={dataset}>
            {dataset}
          </SelectItem>
        ))}
      </Select>
      {/* <p className="text-small text-default-500">Selected: {Array.from(selectedFiles).join(', ')}</p> */}
    </div>
  );
});



const SearchBar = React.memo(({ onSearch }: { onSearch: (query: string) => void }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const handleSearch = () => onSearch(searchQuery);

  return (
    <div className="flex w-full gap-2">
      <input type="text" placeholder="Search by metadata..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSearch()} className="w-full p-2 border rounded-lg" />
      <button onClick={handleSearch} className="px-4 py-2 bg-blue-500 text-white rounded-lg">Search</button>
    </div>
  );
});

export default function ProblemsViewer() {
  const [data, setData] = useState<ProblemData[]>([]);
  const [filteredData, setFilteredData] = useState<ProblemData[]>([]);
  const [currentPage, setCurrentPage] = useState(1);

  const fetchData = useCallback((files: string[]) => {
    if (files.length > 0) {
      const query = files.map(file => `files=${file}`).join('&');
      fetch(`/api/data?${query}`).then(res => res.json()).then((data: ProblemData[]) => {
        setData(data);
        setFilteredData(data);
      });
    } else {
      setData([]);
      setFilteredData([]);
    }
  }, []);

  const handleSearch = useCallback((query: string) => {
    const result = data.filter(item => item.metadata.toLowerCase().includes(query.toLowerCase()));
    setFilteredData(result);
    setCurrentPage(1);
  }, [data]);

  const totalPages = Math.ceil(filteredData.length / ITEMS_PER_PAGE);
  const currentData = filteredData.slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE);

  return (
    <MathJaxContext config={mathJaxConfig}>
      <div className="p-4 space-y-4">
        <h1 className="text-3xl font-extrabold text-center">Problems Viewer</h1>
        <div className="flex space-x-2 mb-4 w-full">
          <FileSelector onSelect={fetchData} />
          <SearchBar onSearch={handleSearch} />
        </div>
        <div className="flex justify-between items-center mt-4">
          <button onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))} disabled={currentPage === 1} className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50">Prev</button>
          <span className="text-lg font-medium">Page {currentPage} of {totalPages}</span>
          <button onClick={() => setCurrentPage((p) => Math.min(p + 1, totalPages))} disabled={currentPage === totalPages} className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50">Next</button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {currentData.map((item, index) => (
            <div key={index} className="p-4 rounded-2xl shadow-md bg-white dark:bg-gray-800">
              <p className="font-bold text-lg mt-2">Metadata:</p><pre>{item.metadata}</pre>
              <p className="font-bold text-lg">Problem:</p><MathJax inline>{item.problem}</MathJax>
              <p className="font-bold text-lg mt-2">Response:</p><MathJax inline>{item.response}</MathJax>
              <p className="font-bold text-lg mt-2">Solution:</p><MathJax inline>{item.solution}</MathJax>
              <p className="font-bold text-lg mt-2">Full Response:</p><MathJax inline>{item.model_response}</MathJax>
            </div>
          ))}
        </div>

      </div>
    </MathJaxContext>
  );
}